# optimal model for analyzing scaled load
# setting up the model
from pyomo.environ import *
import pandas as pd
import time
import json
import random

big_M = 1e6
gas_price_value = 0.287               # price for gas, /MWh
gas_carbon = 0.4                      # gas carbon emission coefficient, unit: tons/MWh
carbon_emission_rate = 0.5366         # grid carbon emission coefficient, unit: tons/MWh

discount_rate = 0.06                  # discount rate

pv_rate = {1: 674.294, 2: 586.419, 3: 579.049, 4: 683.491}  # maximum photovoltaic generation per square kilometer for each city, unit: MW
sc_rate = {1: 3371.47, 2: 2932.09, 3: 2895.25, 4: 3417.45}  # maximum solar thermal generation per square kilometer for each city, unit: MW

#-----------------------------------------------------
# Device class definitions
# Photovoltaic/Solar thermal devices
class solar_device:
    def __init__(self, label, efficiency, output_kind, cost, area_rate):
        self.label = label              # device name
        self.efficiency = efficiency    # list, representing the efficiency of outputs for different energy types
        self.output_kind = output_kind  # list, representing the output energy types
        self.cost = cost                # construction cost per unit input capacity
        self.area_rate = area_rate      # generation/thermal output per unit area, unit: MW/m^2
        pass

# Energy conversion devices
class conversion_device:
    # Energy conversion device
    def __init__(self, label, efficiency, input_kind, output_kind, cost):
        # Record basic information about the device
        self.label = label              # device name
        self.efficiency = efficiency    # list, representing the efficiency of outputs for different energy types
        self.input_kind = input_kind    # string, representing the input energy type (currently supports single input type only)
        self.output_kind = output_kind  # list, representing the output energy types
        self.cost = cost                # construction cost per unit input capacity
        pass

class storage_device:
    # Energy storage device
    def __init__(self, label, input_efficiency, output_efficiency, input_kind,\
        output_kind, cost, self_discharge_rate, t_duration):
        self.label = label                          # device name
        self.input_efficiency = input_efficiency    # input efficiency
        self.output_efficiency = output_efficiency  # output efficiency
        self.input_kind = input_kind                # string, representing the input energy type
        self.output_kind = output_kind              # string, representing the output energy type
        self.cost = cost                            # construction cost per unit storage capacity
        self.self_discharge_rate = self_discharge_rate  # self-discharge rate
        self.t_duration = t_duration                # duration of the storage device cycle
        pass

# read and process the data
#-----------------------------------------------------
def load_scaling(load, scale):
    load_scaled = [i * (1 + scale / 100) for i in load]
    return load_scaled

def read_data(city_code, load, scale):
    # Select the corresponding file name based on the input city code
    city_map = {
        1: "Beijing",
        3: "Wuhan",
        4: "Urumqi"
    }

    # Data file path
    file_path = './data/all_data.json'
    data = json.load(open(file_path, 'r'))
    # Read data
    elec_load = data[city_map[city_code]]["elec"]
    heat_load = data[city_map[city_code]]["heat"]
    cold_load = data[city_map[city_code]]["cool"]
    # uniform distribution as disturbance for Wuhan
    elec_factor = [random.uniform(0.8, 1.2) for _ in range(8760)]
    # Scale the data
    elec_load = [elec_load[i] * elec_factor[i] for i in range(8760)]

    elec_price = data['elec_price']
    gas_price = [gas_price_value for _ in range(8760)]
    pv_I = data[city_map[city_code]]["pv"]
    elec_carbon = [carbon_emission_rate for _ in range(8760)]
    # Scale the data
    if load =='elec':
        elec_load = load_scaling(elec_load, scale)
    elif load == 'heat':
        heat_load = load_scaling(heat_load, scale)
    elif load == 'cool':
        cold_load = load_scaling(cold_load, scale)
    elif load == 'all':
        heat_load = load_scaling(heat_load, scale)
        cold_load = load_scaling(cold_load, scale)
    else:
        pass
    return elec_load, heat_load, cold_load, elec_price, gas_price , pv_I, elec_carbon
# optimization function
# cost distbuted yearly
def old_equal(cost, year):
    return discount_rate/(1 - (1 + discount_rate) ** (-year))*cost

def optimal_plan(city_code, carbon_price, pv_space, load, scale):
    start_time = time.time()
    print('Modeling...')
    #-----------------------------------------------------

    m = ConcreteModel()
    #-----------------------------------------------------
    elec_load, heat_load, cold_load, elec_price, gas_price, pv_I, elec_carbon = read_data(city_code, load, scale)
    m.elec_load = elec_load
    m.heat_load = heat_load
    m.cold_load = cold_load
    m.pv_I = pv_I                   #normalization
    
    m.elec_price = elec_price
    m.gas_price = gas_price
    
    m.elec_carbon = elec_carbon     # the carbon factor propotional to normalized electricity load
    
    m.carbon_price = carbon_price   # price of carbon emission Wrmb/ton
    
    m.pv_space = pv_space           # the space of solar device installation
    
    # 所有待规划设备的字典
    # 光伏发电/太阳能集热器
    pv = solar_device(label = "PV", efficiency=[0.85], output_kind = ["elec"], cost = old_equal(689835, 25), area_rate = 100)
    sc = solar_device(label = "SC", efficiency=[0.75], output_kind = ["heat"], cost = old_equal(131400, 20), area_rate = 500)
    m.solar_device_list = [pv, sc]
    
    # 各能量转换设备信息录入
    heat_elec_collab = conversion_device(label = "CHP", efficiency=[0.3, 0.65], \
                                        input_kind = "gas", output_kind = ["elec", "heat"], cost = old_equal(773.07, 30))

    elec_boiler = conversion_device(label = "Electric_Boiler", efficiency=[0.85], \
                                        input_kind = "elec", output_kind = ["heat"], cost = old_equal(55.48, 20))
    
    compress_cold = conversion_device(label = "CERG", efficiency=[3.5], \
                                        input_kind = "elec", output_kind = ["cold"], cost = old_equal(123.37, 15))
    
    absorb_cold = conversion_device(label = "WARP", efficiency=[1.2], \
                                        input_kind = "heat", output_kind = ["cold"], cost = old_equal(86.14, 20))
    
    gas_boilder = conversion_device(label = "Gas_Boiler", efficiency=[0.85], \
                                        input_kind = "gas", output_kind = ["heat"], cost = old_equal(44, 20))
    
    Ground_source_heat_pump_heat = conversion_device(label = "Ground_Heat_Pump_heat", efficiency=[3.4], \
                                        input_kind = "elec", output_kind = ["heat"], cost = old_equal(330, 20))
    Ground_source_heat_pump_cold = conversion_device(label = "Ground_Heat_Pump_cold", efficiency=[4.6], \
                                        input_kind = "elec", output_kind = ["cold"], cost = 0)
    
    m.conversion_device_list = [heat_elec_collab, elec_boiler,compress_cold,\
                            absorb_cold, gas_boilder, Ground_source_heat_pump_heat,\
                            Ground_source_heat_pump_cold]
    
    #storage devices
    elec_storage = storage_device(label = "Elec_Storage", input_efficiency = 0.9539, output_efficiency = 0.9539, \
                                input_kind = "elec", output_kind = "elec", cost = old_equal(228, 10), self_discharge_rate = 0.00054, t_duration = 2)
    heat_storage = storage_device(label = "Heat_Storage", input_efficiency = 0.894, output_efficiency = 0.894, \
                                input_kind = "heat", output_kind = "heat", cost = old_equal(11, 20), self_discharge_rate = 0.0075,  t_duration = 5)
    cold_storage = storage_device(label = "Cold_Storage", input_efficiency = 0.894, output_efficiency = 0.894, \
                                input_kind = "cold", output_kind = "cold", cost = old_equal(11, 20), self_discharge_rate = 0.0075,  t_duration = 5)
    Seasonal_Heat_Storage = storage_device(label = "Seasonal_Heat_Storage", input_efficiency = 1, output_efficiency = 1, \
                                input_kind = "heat", output_kind = "heat", cost = old_equal(1.1325, 20),  self_discharge_rate = 0, t_duration = 1)
    
    m.storage_device_list = [elec_storage, heat_storage, cold_storage, Seasonal_Heat_Storage]
    
    #-------------------------------------------------------------------
    #defined time sets
    m.t_8760 = Set(initialize = [i for i in range(8760)])
    m.day_364 = Set(initialize = [i for i in range(364)])
    #----------------------------------------
    
    # index sets for conversion and storage devices
    converter_power_index = []
    for device in m.conversion_device_list:
        # input_kind
        device_kind_name = device.label + "_" + device.input_kind
        converter_power_index.append(device_kind_name)
        # output_kind
        for output_kind in device.output_kind:
            device_kind_name = device.label + "_" + output_kind
            converter_power_index.append(device_kind_name)
    
    m.set_converter_power = Set(initialize = converter_power_index)
    m.set_converter = Set(initialize = range(len(m.conversion_device_list)))
    m.set_storage = Set(initialize = range(len(m.storage_device_list)))

    #----------------------------------------
    #converter constrain
    m.convert_invest = Var(m.set_converter, within = NonNegativeReals)
    m.convert_power = Var(m.set_converter_power, m.t_8760, within = NonNegativeReals)


    for device in m.conversion_device_list:
        for output in device.output_kind:
            i = device.output_kind.index(output)
            fun_name = "c_{}_{}".format(device.label, output)
            input_kind = device.label + "_" + device.input_kind
            output_kind = device.label + "_" + output
            code_str = f'def {fun_name}(model, t):\n\t' \
                    f'return {device.efficiency[i]} * model.convert_power["{input_kind}",  t] == model.convert_power["{output_kind}", t]'
            
            local_env = {'m': m}
            exec(code_str, globals(), local_env)
            exec(f"m.constraint_{fun_name} = Constraint(m.t_8760, rule={fun_name})", globals(), local_env)
    
    #investment constraints for conversion devices
    def c_max_converter_input(model, device_id, t):
        m = model
        the_device = m.conversion_device_list[device_id]
        input_key = the_device.label + "_" + the_device.input_kind
        
        return m.convert_power[input_key, t] <= m.convert_invest[device_id]
    m.c_max_converter_input = Constraint(m.set_converter, m.t_8760, rule=c_max_converter_input)
    
    #----------------------------------------
    # binary variables for STES usage
    m.seasonal_heat_storage_used = Var(within=Binary)

    # binary variables for ground source heat pump
    m.Ground_source_heat_pump_heat_flag = Var(m.t_8760, within=Binary)
    m.Ground_source_heat_pump_cold_flag = Var(m.t_8760, within=Binary)
    
    # TODO: set the seasonal heat storage charging and discharging time, change with month
    
    
    
    # heat_pump capacity constraints
    def c_Ground_source_heat_pump(model):
        m = model
        Ground_heat_pump_heat_index = next(i for i, device in enumerate(m.conversion_device_list) if device.label == "Ground_Heat_Pump_heat")
        Ground_heat_pump_cold_index = next(i for i, device in enumerate(m.conversion_device_list) if device.label == "Ground_Heat_Pump_cold")
        return m.convert_invest[Ground_heat_pump_heat_index] == m.convert_invest[Ground_heat_pump_cold_index]
    m.c_Ground_source_heat_pump = Constraint(rule=c_Ground_source_heat_pump)
        
    def c_Ground_source_heat_pump_heat_input(model, t):
        m = model
        return m.convert_power["Ground_Heat_Pump_heat_elec", t] <= m.Ground_source_heat_pump_heat_flag[t] * big_M
    m.c_Ground_source_heat_pump_input = Constraint(m.t_8760, rule=c_Ground_source_heat_pump_heat_input)
    
    def c_Ground_source_heat_pump_cold_input(model, t):
        m = model
        return m.convert_power["Ground_Heat_Pump_cold_elec", t] <= m.Ground_source_heat_pump_cold_flag[t] * big_M
    m.c_Ground_source_heat_pump_output = Constraint(m.t_8760, rule=c_Ground_source_heat_pump_cold_input)
    
    def c_Ground_source_heat_pump_mutual_exclusive(model, t):
        m = model
        return m.Ground_source_heat_pump_heat_flag[t] + m.Ground_source_heat_pump_cold_flag[t] <= 1
    m.c_Ground_source_heat_pump_mutual_exclusive = Constraint(m.t_8760, rule=c_Ground_source_heat_pump_mutual_exclusive)

#----------------------------------------
    m.storage_invest = Var(m.set_storage, within = NonNegativeReals)
    m.storage_charge_power = Var(m.set_storage,  m.t_8760, within = NonNegativeReals)
    m.storage_discharge_power = Var(m.set_storage,  m.t_8760, within = NonNegativeReals)
    m.storage_soc = Var(m.set_storage, m.t_8760, within = NonNegativeReals)

    # STES connection with heat pump
    m.seasonal_heat_storage_heating = Var(m.t_8760, within = NonNegativeReals)
    
    # cooling
    def c_seasonal_heat_storage_input(model, t):
        m = model
        seasonal_heat_storage_index = next(i for i, device in enumerate(m.storage_device_list) if device.label == "Seasonal_Heat_Storage")
        return m.storage_charge_power[seasonal_heat_storage_index, t] == (4.6 + 1) * m.convert_power["Ground_Heat_Pump_cold_elec", t] 
    m.c_seasonal_heat_storage_input = Constraint(m.t_8760, rule=c_seasonal_heat_storage_input)

    # heating
    def c_seasonal_heat_storage_output(model, t):
        m = model
        seasonal_heat_storage_index = next(i for i, device in enumerate(m.storage_device_list) if device.label == "Seasonal_Heat_Storage")
        return m.storage_discharge_power[seasonal_heat_storage_index, t] == (3.4 - 1) * m.convert_power["Ground_Heat_Pump_heat_elec", t]
    m.c_seasonal_heat_storage_output = Constraint(m.t_8760, rule=c_seasonal_heat_storage_output)

    # storage transfer constraints
    def c_storage_transfer(model, storage_id, t):
        m = model
        the_device = m.storage_device_list[storage_id]
        
        if the_device.label == "Seasonal_Heat_Storage":
            if t == 0:
                return m.storage_soc[storage_id, 0] == m.storage_soc[storage_id, 8759] * (1 - the_device.self_discharge_rate)  \
                    + m.storage_charge_power[storage_id, 0] * the_device.input_efficiency \
                    - m.storage_discharge_power[storage_id, 0] / the_device.output_efficiency \
                    + m.seasonal_heat_storage_heating[0]
            else:
                return m.storage_soc[storage_id, t] == m.storage_soc[storage_id, t-1] * (1 - the_device.self_discharge_rate) \
                    + m.storage_charge_power[storage_id, t] * the_device.input_efficiency \
                    - m.storage_discharge_power[storage_id, t] / the_device.output_efficiency \
                    + m.seasonal_heat_storage_heating[t]
        else:
            if t == 0:
                return m.storage_soc[storage_id, 0] == m.storage_soc[storage_id, 8759] * (1 - the_device.self_discharge_rate)  \
                    + m.storage_charge_power[storage_id, 0] * the_device.input_efficiency \
                    - m.storage_discharge_power[storage_id, 0] / the_device.output_efficiency 
            else:
                return m.storage_soc[storage_id, t] == m.storage_soc[storage_id, t-1] * (1 - the_device.self_discharge_rate) \
                    + m.storage_charge_power[storage_id, t] * the_device.input_efficiency \
                    - m.storage_discharge_power[storage_id, t] / the_device.output_efficiency 
    m.c_storage_transfer = Constraint(m.set_storage, m.t_8760, rule = c_storage_transfer)
    
    def c_storage_daily_cycle(model, storage_id, day):
        m = model
        the_device = m.storage_device_list[storage_id]
        if the_device.label == "Seasonal_Heat_Storage":
            return Constraint.Skip
        else:
            return m.storage_soc[storage_id, day * 24] == m.storage_soc[storage_id, (day + 1) * 24]
    m.c_storage_daily_cycle = Constraint(m.set_storage, m.day_364, rule = c_storage_daily_cycle)

    def c_max_storage_input(model, storage_id, t):
        m = model
        the_device = m.storage_device_list[storage_id]

        if the_device.label == "Seasonal_Heat_Storage":
            return Constraint.Skip
        else:
            return m.storage_charge_power[storage_id, t] <= m.storage_invest[storage_id] / the_device.t_duration
    m.c_max_storage_input = Constraint(m.set_storage, m.t_8760, rule=c_max_storage_input) 
    
    
    def c_max_storage_output(model, storage_id,  t):
        m = model
        the_device = m.storage_device_list[storage_id]

        if the_device.label == "Seasonal_Heat_Storage":
            return Constraint.Skip
        else:
            return m.storage_discharge_power[storage_id, t] <= m.storage_invest[storage_id] / the_device.t_duration
    m.c_max_storage_output = Constraint(m.set_storage, m.t_8760, rule = c_max_storage_output)

    
    def c_max_storage_cap(model, storage_id,  t):
        m = model
        return m.storage_soc[storage_id, t] <= m.storage_invest[storage_id]
    m.c_max_storage_cap = Constraint(m.set_storage, m.t_8760, rule = c_max_storage_cap)
    
    def c_seasonal_heat_storage(model):
        m = model
        for storage_id, device in enumerate(m.storage_device_list):
            if device.label == "Seasonal_Heat_Storage":
                return m.storage_invest[storage_id] <= big_M * m.seasonal_heat_storage_used
        return Constraint.Skip
    m.c_seasonal_heat_storage = Constraint(rule=c_seasonal_heat_storage)

    #----------------------------------------
    m.solar_Area = Var(['pv', 'sc'], within=NonNegativeReals, initialize=0)
    def c_solar_invest(model):
        m = model
        return m.solar_Area['pv'] + m.solar_Area['sc'] <= m.pv_space
    m.c_solar_invest = Constraint(rule=c_solar_invest)
    
    # energy bus balance
    m.set_energy_kind = Set(initialize=["elec", "heat", "cold", "gas"])
    m.set_buy_energy = Set(initialize=["elec", "gas"])
    m.set_solar_energy = Set(initialize=["elec", "heat"])
    # electricity from grid
    m.buy_energy = Var(m.set_buy_energy, m.t_8760, within = NonNegativeReals)
    # solar energy production
    m.solar_energy = Var(m.set_solar_energy, m.t_8760, within=NonNegativeReals)
    def c_solar_energy_elec(model, t):
        m = model
        return m.solar_energy["elec", t] <= m.solar_Area['pv'] * m.pv_I[t] * pv_rate[city_code]

    def c_solar_energy_heat(model, t):
        m = model
        return m.solar_energy["heat", t] <= m.solar_Area['sc'] * m.pv_I[t] * sc_rate[city_code]

    m.c_solar_energy_elec = Constraint(m.t_8760, rule=c_solar_energy_elec)
    m.c_solar_energy_heat = Constraint(m.t_8760, rule=c_solar_energy_heat)

    def c_bus_balance(model, energy_kind, t):
        m = model
        total_output = 0
        if energy_kind == "elec" or energy_kind == "gas":
            total_output += m.buy_energy[energy_kind, t]
        if energy_kind == "elec":
            total_output += m.solar_energy["elec", t]
        elif energy_kind == "heat":
            total_output += m.solar_energy["heat", t]
        
        for device in m.conversion_device_list:
            if energy_kind == device.input_kind:
                input_key = device.label + "_" + energy_kind
                total_output -= m.convert_power[input_key, t]
            elif energy_kind in device.output_kind:
                output_key = device.label + "_" + energy_kind
                total_output += m.convert_power[output_key, t]
        
        for device_id in range(len(m.storage_device_list)):
            the_device = m.storage_device_list[device_id]
            if the_device.label == "Seasonal_Heat_Storage":
                continue
            if energy_kind == the_device.input_kind:
                total_output += m.storage_discharge_power[device_id, t] - m.storage_charge_power[device_id, t]
        
        if energy_kind == "elec":
            return total_output == m.elec_load[t]
        elif energy_kind == "heat":
            return total_output == m.heat_load[t] + m.seasonal_heat_storage_heating[t]
        elif energy_kind == "cold":
            return total_output == m.cold_load[t]
        elif energy_kind == "gas":
            return total_output == 0
        
    m.c_bus_balance = Constraint(m.set_energy_kind, m.t_8760, rule=c_bus_balance)
    #----------------------------------------
    # total cost
    m.total_cost = Var(within = NonNegativeReals)
    def c_objective(model):
        m = model
        #------------------------------------------------
        # investment cost
        m.invest_cost = 0
        for device_id in range(len(m.conversion_device_list)):
            the_device = m.conversion_device_list[device_id]
            m.invest_cost += the_device.cost * m.convert_invest[device_id]
        for device_id in range(len(m.storage_device_list)):
            the_device = m.storage_device_list[device_id]
            if the_device.label == "Seasonal_Heat_Storage":
                m.invest_cost += old_equal(116.32436, 20) * m.seasonal_heat_storage_used + the_device.cost * m.storage_invest[device_id]
            else:
                m.invest_cost += the_device.cost * m.storage_invest[device_id]
        m.invest_cost += pv.cost * m.solar_Area['pv']
        m.invest_cost += sc.cost * m.solar_Area['sc']
        #------------------------------------------------
        #operation cost
        m.oper_cost = 0
        m.carbon_cost = 0
        for t in m.t_8760:
            # unit as WRMB
            m.oper_cost += m.elec_price[t] * m.buy_energy["elec", t] / 10000
            m.oper_cost += m.gas_price[t] * m.buy_energy["gas", t] / 10000
            # carbin enmssion cost
            m.carbon_cost += m.carbon_price * m.elec_carbon[t] * m.buy_energy["elec", t] / 10000
            m.carbon_cost += m.carbon_price * gas_carbon * m.buy_energy["gas", t] / 10000

        # yearly cost
        return m.total_cost == m.invest_cost + m.oper_cost + m.carbon_cost
    m.c_objective = Constraint(rule = c_objective)
    #----------------------------------------
    m.objective = Objective(expr = m.total_cost, sense = minimize)

    #----------------------------------------
    print('Construction Complete. {}s used. Optimizing...'.format(time.time()-start_time))
    start_time = time.time()
    opt = SolverFactory('gurobi') 
    opt.solve(m)
    print('Optimization Complete. {}s used.'.format(time.time()-start_time))

    return m

