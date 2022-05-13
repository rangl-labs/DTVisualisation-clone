import logging
from pathlib import Path

import pandas as pd
import numpy as np
import gym
# import reference_environment

# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)
# logger.setLevel(logging.DEBUG)

def run_middleware(policy_configs):
    
    # Create an environment named env
    env = gym.make("reference_environment:rangl-NBMdata-v0")
    # use a fixed seed to guarantee reproducibility for same policy config 
    seed = 123456
    env.seed(seed)
    
    # Generate a random action and check it has the right length
    # action = env.action_space.sample()
    # assert len(action) == 3
    
    # Reset the environment
    env.reset()
    # Check the to_observation method
    # assert len(env.observation_space.sample()) == len(env.state.to_observation())
    done = False
    
    # use a fixed seed to guarantee reproducibility for same policy config 
    np.random.seed(seed)
    
    target_year = 2070
    total_years = target_year - 2022 + 1
    # env.set_steps_per_episode(total_years)
    
    # Policies Configuration: High means using only 1/3 of the total_years to finish installation to all buildings, i.e., fastest
    # Medium means using 2/3 of the total_years to finish installation to all buildings,
    # Low means using all total_years to finish installation to all buildings, i.e., not finished until the target year
    # policies = np.array(['H','M','L','L'])
    # convert Policies Configuration with target year to percentage of buildings getting new tech installed per year:
    # action = np.zeros(len(env.action_space.sample()))
    # for i in np.arange(len(action)):
    #     if policies[i] == 'H':
    #         action[i] = 1.0/(total_years/3.0)
    #     elif policies[i] == 'M':
    #         action[i] = 1.0/(total_years*2/3.0)
    #     else:
    #         action[i] = 1.0/total_years
    # [percentage of buildings get new tech #1 deployed, percentage of buildings get new tech #2 deployed, percentage of buildings get new tech #3 deployed]
    # action = [0.1, 0.2, 0.3]
    # New approach: using the relation table to convert 3x3 policy configs to relative intensities of the 3 techs, as in
    # https://github.com/moriartyjm/DTVisualisation/issues/27#issuecomment-1095726680
    # order of techs: [Heat pumps, Solar PV, External wall Insulation, EV charge points]
    policy_tech_relation = np.array([[0.8,0.5,0],[0,0.3,0],[0.2,0,0.9],[0,0.2,0.1]]) # the 4x3 table in the GitHub comment above, with the above order or techs
    min_intensities = np.sum(policy_tech_relation, axis=1)
    max_intensities = np.sum(policy_tech_relation*3, axis=1)
    earliest_finishing_year = np.ones(len(policy_tech_relation))*2035
    latest_finishing_year = np.ones(len(policy_tech_relation))*2065
    # policy_configs = np.array(['L','L','L']) # ['thermal comfort', 'mass electrification', 'energy resilience']
    # policy_configs[0] = input("Enter the policy config for Thermal Comfort (input 'L' for low, 'M' for medium, 'H' for high): ")
    # policy_configs[1] = input("Enter the policy config for Mass Electrification (input 'L' for low, 'M' for medium, 'H' for high): ")
    # policy_configs[2] = input("Enter the policy config for Energy Resilience (input 'L' for low, 'M' for medium, 'H' for high): ")
    tech_intensities = policy_tech_relation*1.0
    for i in np.arange(len(policy_configs)):
        if policy_configs[i].upper() == 'M':
            tech_intensities[:,i] = tech_intensities[:,i]*2.0
        elif policy_configs[i].upper() == 'H':
            tech_intensities[:,i] = tech_intensities[:,i]*3.0
    tech_intensities = np.sum(tech_intensities, axis = 1)
    action = np.zeros(len(tech_intensities))
    for i in np.arange(len(action)):
        action[i] = 1.0/(latest_finishing_year[i] 
                         + (tech_intensities[i]-min_intensities[i])*(earliest_finishing_year[i]-latest_finishing_year[i])
                         /(max_intensities[i]-min_intensities[i])
                         - 2022)
    
    
    # weighting parameters to map from 1672 buildings in NBM data to 374 local authority districts using some weighted average:
    # # currently these parameters are randomly generated:
    # weights_UKareaID = np.random.rand(len(env.param.serialanon), 374)
    # # normalize each column:
    # weights_UKareaID = weights_UKareaID/np.sum(weights_UKareaID, axis = 0)
    # revised weights, equivalent to randomly choosing some number of buildings as belonging to each of the 374 local districts,
    # and then set a same weight = 1/# of buildings in each AreaID:
    weights_UKareaID = np.zeros((len(env.param.serialanon), 374))
    NumBldg_perUKareaID = int(len(env.param.serialanon)*0.02)
    for i in np.arange(weights_UKareaID.shape[-1]):
        weights_UKareaID[np.random.choice(np.arange(len(weights_UKareaID)), NumBldg_perUKareaID, replace=False),i] = 1.0/NumBldg_perUKareaID
    weights_ManAreaID = np.zeros((len(env.param.serialanon), 282))
    NumBldg_perManAreaID = int(len(env.param.serialanon)*0.02)
    for i in np.arange(weights_ManAreaID.shape[-1]):
        weights_ManAreaID[np.random.choice(np.arange(len(weights_ManAreaID)), NumBldg_perManAreaID, replace=False),i] = 1.0/NumBldg_perManAreaID
    Local39BldgSelectionTF = np.full(len(env.param.serialanon), False)
    Local39BldgSelectionTF[np.random.choice(np.arange(len(Local39BldgSelectionTF)), 39, replace=False)] = True
    
    avePctPump_UKareaID = []
    avePctSolar_UKareaID = []
    avePctInsulation_UKareaID = []
    avePctEV_UKareaID = []
    aveEER_UKareaID = []
    aveElecUse_UKareaID = []
    aveGasUse_UKareaID = []
    aveEmissions_UKareaID = []
    
    avePctPump_ManAreaID = []
    avePctSolar_ManAreaID = []
    avePctInsulation_ManAreaID = []
    avePctEV_ManAreaID = []
    aveEER_ManAreaID = []
    aveElecUse_ManAreaID = []
    aveGasUse_ManAreaID = []
    aveEmissions_ManAreaID = []
    
    local39BldgEER = []
    local39BldgElecUse = []
    local39BldgGasUse = []
    local39BldgEmissions = []
    local39BldgFinishingYears = np.full((39,4), target_year + 1)
    
    maxNumBldgUK = 25000000.0 # According to https://www.statista.com/statistics/232302/number-of-dwellings-in-england/ there are approx. 25 million homes in the UK, as discussed in https://github.com/moriartyjm/DTVisualisation/issues/27#issuecomment-1115952033
    aveEVchargingPointPowerMW = 0.0072 # 7.2 kW = 0.0072 Mega Wats, as discussed in https://github.com/moriartyjm/DTVisualisation/issues/27#issuecomment-1120257736
    techSummary = np.zeros((8,total_years))
    
    while not done:
        # Specify the action. Check the effect of any fixed policy by specifying the action here:
        observation, reward, done, _ = env.step(action)
        # logger.debug(f"step_count: {env.state.step_count}")
        # logger.debug(f"action: {action}")
        # logger.debug(f"observation: {observation}")
        # logger.debug(f"reward: {reward}")
        # logger.debug(f"done: {done}")
        # print()
        # env.action_space.sample()
        # env.state.currentNBMtable.to_csv('../Visualization_Data/NBM_selected_columns_policies' + ''.join(policies) + '_year' + str(env.state.step_count+2022) + '.csv')
        avePctPump_UKareaID.append(env.state.deployedTF[:,0]*1 @ weights_UKareaID)
        avePctSolar_UKareaID.append(env.state.deployedTF[:,1]*1 @ weights_UKareaID)
        avePctInsulation_UKareaID.append(env.state.deployedTF[:,2]*1 @ weights_UKareaID)
        avePctEV_UKareaID.append(env.state.deployedTF[:,3]*1 @ weights_UKareaID)
        aveEER_UKareaID.append(env.state.currentNBMtable1RowPerBldg["sap/sap_rating"] @ weights_UKareaID)
        aveElecUse_UKareaID.append(env.state.currentNBMtable1RowPerBldg["Electricity use (60% of total kWh/year)"] @ weights_UKareaID)
        aveGasUse_UKareaID.append(env.state.currentNBMtable1RowPerBldg["Gas use (40% of total kWh/year)"] @ weights_UKareaID)
        aveEmissions_UKareaID.append(env.state.currentNBMtable1RowPerBldg["Total CO2 emissions (kg/year)"] @ weights_UKareaID)
        
        avePctPump_ManAreaID.append(env.state.deployedTF[:,0]*1 @ weights_ManAreaID)
        avePctSolar_ManAreaID.append(env.state.deployedTF[:,1]*1 @ weights_ManAreaID)
        avePctInsulation_ManAreaID.append(env.state.deployedTF[:,2]*1 @ weights_ManAreaID)
        avePctEV_ManAreaID.append(env.state.deployedTF[:,3]*1 @ weights_ManAreaID)
        aveEER_ManAreaID.append(env.state.currentNBMtable1RowPerBldg["sap/sap_rating"] @ weights_ManAreaID)
        aveElecUse_ManAreaID.append(env.state.currentNBMtable1RowPerBldg["Electricity use (60% of total kWh/year)"] @ weights_ManAreaID)
        aveGasUse_ManAreaID.append(env.state.currentNBMtable1RowPerBldg["Gas use (40% of total kWh/year)"] @ weights_ManAreaID)
        aveEmissions_ManAreaID.append(env.state.currentNBMtable1RowPerBldg["Total CO2 emissions (kg/year)"] @ weights_ManAreaID)
        
        local39BldgEER.append(env.state.currentNBMtable1RowPerBldg["sap/sap_rating"].to_numpy()[Local39BldgSelectionTF]/100.0)
        local39BldgElecUse.append(env.state.currentNBMtable1RowPerBldg["Electricity use (60% of total kWh/year)"].to_numpy()[Local39BldgSelectionTF])
        local39BldgGasUse.append(env.state.currentNBMtable1RowPerBldg["Gas use (40% of total kWh/year)"].to_numpy()[Local39BldgSelectionTF])
        local39BldgEmissions.append(env.state.currentNBMtable1RowPerBldg["Total CO2 emissions (kg/year)"].to_numpy()[Local39BldgSelectionTF]/1000.0)
        
        for i in np.arange(local39BldgFinishingYears.shape[-1]):
            local39BldgFinishingYears[
                np.logical_and(local39BldgFinishingYears[:,i] == target_year+1, env.state.deployedTF[Local39BldgSelectionTF,i]),i
                ] = env.state.step_count + 2022
        
        techSummary[0,env.state.step_count] = int(np.sum(env.state.deployedTF[:,1])*maxNumBldgUK/env.state.deployedTF.shape[0])
        techSummary[1,env.state.step_count] = np.sum(env.state.currentNBMtable1RowPerBldg["renewables/pv_peakpower_front"]/1000)*maxNumBldgUK/env.state.deployedTF.shape[0]
        techSummary[2,env.state.step_count] = int(np.sum(env.state.deployedTF[:,0])*maxNumBldgUK/env.state.deployedTF.shape[0])
        techSummary[3,env.state.step_count] = np.sum(env.state.currentNBMtable1RowPerBldg["Installed heat pumps power in Megawatts"])*maxNumBldgUK/env.state.deployedTF.shape[0]
        techSummary[4,env.state.step_count] = int(np.sum(env.state.deployedTF[:,3])*maxNumBldgUK/env.state.deployedTF.shape[0])
        techSummary[5,env.state.step_count] = np.sum(env.state.deployedTF[:,3])*aveEVchargingPointPowerMW*maxNumBldgUK/env.state.deployedTF.shape[0]
        techSummary[6,env.state.step_count] = int(np.sum(env.state.deployedTF[:,2])*maxNumBldgUK/env.state.deployedTF.shape[0])
        techSummary[7,env.state.step_count] = np.sum(env.state.currentNBMtable1RowPerBldg["Total wall insulation area (m^2)"])/1e+6*maxNumBldgUK/env.state.deployedTF.shape[0]
    
    avePctPump_UKareaID = np.array(avePctPump_UKareaID).T
    avePctSolar_UKareaID = np.array(avePctSolar_UKareaID).T
    avePctInsulation_UKareaID = np.array(avePctInsulation_UKareaID).T
    avePctEV_UKareaID = np.array(avePctEV_UKareaID).T
    aveEER_UKareaID = np.array(aveEER_UKareaID)
    aveElecUse_UKareaID = np.array(aveElecUse_UKareaID)
    aveGasUse_UKareaID = np.array(aveGasUse_UKareaID)
    aveEmissions_UKareaID = np.array(aveEmissions_UKareaID)
    aveEER_UKareaID_BestVal = np.amax(aveEER_UKareaID, axis=0)
    aveElecUse_UKareaID_BestVal = np.amin(aveElecUse_UKareaID, axis=0)
    aveGasUse_UKareaID_BestVal = np.amin(aveGasUse_UKareaID, axis=0)
    aveEmissions_UKareaID_BestVal = np.amin(aveEmissions_UKareaID, axis=0)
    aveEER_UKareaID = ((aveEER_UKareaID - np.amin(aveEER_UKareaID, axis=0))/np.ptp(aveEER_UKareaID, axis=0)).T
    aveElecUse_UKareaID = ((aveElecUse_UKareaID - np.amin(aveElecUse_UKareaID, axis=0))/np.ptp(aveElecUse_UKareaID, axis=0)).T
    aveGasUse_UKareaID = ((aveGasUse_UKareaID - np.amin(aveGasUse_UKareaID, axis=0))/np.ptp(aveGasUse_UKareaID, axis=0)).T
    aveEmissions_UKareaID = ((aveEmissions_UKareaID - np.amin(aveEmissions_UKareaID, axis=0))/np.ptp(aveEmissions_UKareaID, axis=0)).T
    
    avePctPump_ManAreaID = np.array(avePctPump_ManAreaID).T
    avePctSolar_ManAreaID = np.array(avePctSolar_ManAreaID).T
    avePctInsulation_ManAreaID = np.array(avePctInsulation_ManAreaID).T
    avePctEV_ManAreaID = np.array(avePctEV_ManAreaID).T
    aveEER_ManAreaID = np.array(aveEER_ManAreaID)
    aveElecUse_ManAreaID = np.array(aveElecUse_ManAreaID)
    aveGasUse_ManAreaID = np.array(aveGasUse_ManAreaID)
    aveEmissions_ManAreaID = np.array(aveEmissions_ManAreaID)
    aveEER_ManAreaID_BestVal = np.amax(aveEER_ManAreaID, axis=0)
    aveElecUse_ManAreaID_BestVal = np.amin(aveElecUse_ManAreaID, axis=0)
    aveGasUse_ManAreaID_BestVal = np.amin(aveGasUse_ManAreaID, axis=0)
    aveEmissions_ManAreaID_BestVal = np.amin(aveEmissions_ManAreaID, axis=0)
    aveEER_ManAreaID = ((aveEER_ManAreaID - np.amin(aveEER_ManAreaID, axis=0))/np.ptp(aveEER_ManAreaID, axis=0)).T
    aveElecUse_ManAreaID = ((aveElecUse_ManAreaID - np.amin(aveElecUse_ManAreaID, axis=0))/np.ptp(aveElecUse_ManAreaID, axis=0)).T
    aveGasUse_ManAreaID = ((aveGasUse_ManAreaID - np.amin(aveGasUse_ManAreaID, axis=0))/np.ptp(aveGasUse_ManAreaID, axis=0)).T
    aveEmissions_ManAreaID = ((aveEmissions_ManAreaID - np.amin(aveEmissions_ManAreaID, axis=0))/np.ptp(aveEmissions_ManAreaID, axis=0)).T
    
    local39BldgEER = np.array(local39BldgEER).T
    local39BldgElecUse = np.array(local39BldgElecUse).T
    local39BldgGasUse = np.array(local39BldgGasUse).T
    local39BldgEmissions = np.array(local39BldgEmissions).T
    local39BldgEER_BestVal = np.amax(local39BldgEER, axis=1)
    local39BldgElecUse_BestVal = np.amin(local39BldgElecUse, axis=1)
    local39BldgGasUse_BestVal = np.amin(local39BldgGasUse, axis=1)
    local39BldgEmissions_BestVal = np.amin(local39BldgEmissions, axis=1)
    
    local39Bldg = np.array(list(zip(local39BldgEER, local39BldgElecUse, local39BldgGasUse, local39BldgEmissions))).reshape(local39BldgEER.shape[0],local39BldgEER.shape[1],4)


    # export to .csv
    # pd.concat([pd.Series(np.arange(374)+1, name = 'AreaID'), 
    #            pd.DataFrame(avePctPump_UKareaID, columns = np.arange(2022,target_year+1))], axis=1).to_csv(
    #                '../Visualization_Data/HeatPumps_weighted_percentages_UKareaIDbyYear_' + ''.join(policy_configs) + '.csv', index=False)
    # pd.concat([pd.Series(np.arange(374)+1, name = 'AreaID'), 
    #            pd.DataFrame(avePctSolar_UKareaID, columns = np.arange(2022,target_year+1))], axis=1).to_csv(
    #                '../Visualization_Data/Solar_weighted_percentages_UKareaIDbyYear_' + ''.join(policy_configs) + '.csv', index=False)
    # pd.concat([pd.Series(np.arange(374)+1, name = 'AreaID'), 
    #            pd.DataFrame(avePctInsulation_UKareaID, columns = np.arange(2022,target_year+1))], axis=1).to_csv(
    #                '../Visualization_Data/Insulation_weighted_percentages_UKareaIDbyYear_' + ''.join(policy_configs) + '.csv', index=False)
    # pd.concat([pd.Series(np.arange(374)+1, name = 'AreaID'), 
    #            pd.DataFrame(avePctEV_UKareaID, columns = np.arange(2022,target_year+1))], axis=1).to_csv(
    #                '../Visualization_Data/EV_weighted_percentages_UKareaIDbyYear_' + ''.join(policy_configs) + '.csv', index=False)
    pd.concat([pd.Series(np.arange(374)+1, name = 'AreaID'), 
                pd.DataFrame(avePctPump_UKareaID, columns = np.arange(2022,target_year+1))], axis=1).to_csv(
                    '../Visualization_Data/MiddlewareGeneratedOutputs/UK/HP/UK-HP-' + ''.join(policy_configs) + '.csv', index=False)
    pd.concat([pd.Series(np.arange(374)+1, name = 'AreaID'), 
                pd.DataFrame(avePctSolar_UKareaID, columns = np.arange(2022,target_year+1))], axis=1).to_csv(
                    '../Visualization_Data/MiddlewareGeneratedOutputs/UK/Solar/UK-Solar-' + ''.join(policy_configs) + '.csv', index=False)
    pd.concat([pd.Series(np.arange(374)+1, name = 'AreaID'), 
                pd.DataFrame(avePctInsulation_UKareaID, columns = np.arange(2022,target_year+1))], axis=1).to_csv(
                    '../Visualization_Data/MiddlewareGeneratedOutputs/UK/Insulation/UK-Insulation-' + ''.join(policy_configs) + '.csv', index=False)
    pd.concat([pd.Series(np.arange(374)+1, name = 'AreaID'), 
                pd.DataFrame(avePctEV_UKareaID, columns = np.arange(2022,target_year+1))], axis=1).to_csv(
                    '../Visualization_Data/MiddlewareGeneratedOutputs/UK/EV/UK-EV-' + ''.join(policy_configs) + '.csv', index=False)
    pd.concat([pd.Series(np.arange(374)+1, name = 'AreaID'), 
                pd.DataFrame(aveEER_UKareaID, columns = np.arange(2022,target_year+1))], axis=1).to_csv(
                    '../Visualization_Data/MiddlewareGeneratedOutputs/UK/EER/UK-EER-' + ''.join(policy_configs) + '.csv', index=False)
    pd.concat([pd.Series(np.arange(374)+1, name = 'AreaID'), 
                pd.DataFrame(aveElecUse_UKareaID, columns = np.arange(2022,target_year+1))], axis=1).to_csv(
                    '../Visualization_Data/MiddlewareGeneratedOutputs/UK/Elec/UK-Elec-' + ''.join(policy_configs) + '.csv', index=False)
    pd.concat([pd.Series(np.arange(374)+1, name = 'AreaID'), 
                pd.DataFrame(aveGasUse_UKareaID, columns = np.arange(2022,target_year+1))], axis=1).to_csv(
                    '../Visualization_Data/MiddlewareGeneratedOutputs/UK/Gas/UK-Gas-' + ''.join(policy_configs) + '.csv', index=False)
    pd.concat([pd.Series(np.arange(374)+1, name = 'AreaID'), 
                pd.DataFrame(aveEmissions_UKareaID, columns = np.arange(2022,target_year+1))], axis=1).to_csv(
                    '../Visualization_Data/MiddlewareGeneratedOutputs/UK/Emission/UK-Emission-' + ''.join(policy_configs) + '.csv', index=False)
    
    pd.concat([pd.Series(np.arange(282)+1, name = 'AreaID'), 
                pd.DataFrame(avePctPump_ManAreaID, columns = np.arange(2022,target_year+1))], axis=1).to_csv(
                    '../Visualization_Data/MiddlewareGeneratedOutputs/Manchester/HeatPumps/Manchester-HP-' + ''.join(policy_configs) + '.csv', index=False)
    pd.concat([pd.Series(np.arange(282)+1, name = 'AreaID'), 
                pd.DataFrame(avePctSolar_ManAreaID, columns = np.arange(2022,target_year+1))], axis=1).to_csv(
                    '../Visualization_Data/MiddlewareGeneratedOutputs/Manchester/Solar/Manchester-Solar-' + ''.join(policy_configs) + '.csv', index=False)
    pd.concat([pd.Series(np.arange(282)+1, name = 'AreaID'), 
                pd.DataFrame(avePctInsulation_ManAreaID, columns = np.arange(2022,target_year+1))], axis=1).to_csv(
                    '../Visualization_Data/MiddlewareGeneratedOutputs/Manchester/Insulation/Manchester-Insulation-' + ''.join(policy_configs) + '.csv', index=False)
    pd.concat([pd.Series(np.arange(282)+1, name = 'AreaID'), 
                pd.DataFrame(avePctEV_ManAreaID, columns = np.arange(2022,target_year+1))], axis=1).to_csv(
                    '../Visualization_Data/MiddlewareGeneratedOutputs/Manchester/EV/Manchester-EV-' + ''.join(policy_configs) + '.csv', index=False)
    pd.concat([pd.Series(np.arange(282)+1, name = 'AreaID'), 
                pd.DataFrame(aveEER_ManAreaID, columns = np.arange(2022,target_year+1))], axis=1).to_csv(
                    '../Visualization_Data/MiddlewareGeneratedOutputs/Manchester/EER/Manchester-EER-' + ''.join(policy_configs) + '.csv', index=False)
    pd.concat([pd.Series(np.arange(282)+1, name = 'AreaID'), 
                pd.DataFrame(aveElecUse_ManAreaID, columns = np.arange(2022,target_year+1))], axis=1).to_csv(
                    '../Visualization_Data/MiddlewareGeneratedOutputs/Manchester/Elec/Manchester-Elec-' + ''.join(policy_configs) + '.csv', index=False)
    pd.concat([pd.Series(np.arange(282)+1, name = 'AreaID'), 
                pd.DataFrame(aveGasUse_ManAreaID, columns = np.arange(2022,target_year+1))], axis=1).to_csv(
                    '../Visualization_Data/MiddlewareGeneratedOutputs/Manchester/Gas/Manchester-Gas-' + ''.join(policy_configs) + '.csv', index=False)
    pd.concat([pd.Series(np.arange(282)+1, name = 'AreaID'), 
                pd.DataFrame(aveEmissions_ManAreaID, columns = np.arange(2022,target_year+1))], axis=1).to_csv(
                    '../Visualization_Data/MiddlewareGeneratedOutputs/Manchester/Emission/Manchester-Emission-' + ''.join(policy_configs) + '.csv', index=False)
    
    pd.concat([pd.Series(np.array(['Solar1','Solar2','HP1','HP2','EV1','EV2','Insulation1','Insulation2']), name = 'TechID'), 
                pd.DataFrame(techSummary, columns = np.arange(2022,target_year+1))], axis=1).to_csv(
                    '../Visualization_Data/MiddlewareGeneratedOutputs/Technology/Technology-' + ''.join(policy_configs) + '.csv', index=False)
    
    
    pd.concat([pd.Series(np.tile(np.arange(39)+1, 4), name = 'BuildingID'), 
                pd.DataFrame(np.vstack((local39BldgEER, local39BldgElecUse, local39BldgGasUse, local39BldgEmissions)), 
                             columns = np.arange(2022,target_year+1))], axis=1).to_csv(
                    '../Visualization_Data/MiddlewareGeneratedOutputs/LocalView/Metrics/LocalView-' + ''.join(policy_configs) + '.csv', index=False)
    
    pd.concat([pd.Series(np.arange(39)+1, name = 'BuildingID'), 
                pd.DataFrame(local39BldgFinishingYears, 
                             columns = np.array(['HP', 'Solar', 'Insulation', 'EV']))], axis=1).to_csv(
                    '../Visualization_Data/MiddlewareGeneratedOutputs/LocalView/Technology/LocalView-' + ''.join(policy_configs) + '.csv', index=False)
    
    
    # pd.concat([pd.Series(np.arange(374)+1, name = 'AreaID'), 
    #            pd.Series(aveEER_UKareaID_BestVal, name = 'Best Value (Max for EER; Min for Elec/Gas/Emissions)')], axis=1).to_csv(
    #                 '../Visualization_Data/MiddlewareGeneratedOutputs/UK/EER/UK-EER-BestValue.csv', index=False)
    # pd.concat([pd.Series(np.arange(374)+1, name = 'AreaID'), 
    #            pd.Series(aveElecUse_UKareaID_BestVal, name = 'Best Value (Max for EER; Min for Elec/Gas/Emissions)')], axis=1).to_csv(
    #                 '../Visualization_Data/MiddlewareGeneratedOutputs/UK/Elec/UK-Elec-BestValue.csv', index=False)
    # pd.concat([pd.Series(np.arange(374)+1, name = 'AreaID'), 
    #            pd.Series(aveGasUse_UKareaID_BestVal, name = 'Best Value (Max for EER; Min for Elec/Gas/Emissions)')], axis=1).to_csv(
    #                 '../Visualization_Data/MiddlewareGeneratedOutputs/UK/Gas/UK-Gas-BestValue.csv', index=False)
    # pd.concat([pd.Series(np.arange(374)+1, name = 'AreaID'), 
    #            pd.Series(aveEmissions_UKareaID_BestVal, name = 'Best Value (Max for EER; Min for Elec/Gas/Emissions)')], axis=1).to_csv(
    #                 '../Visualization_Data/MiddlewareGeneratedOutputs/UK/Emission/UK-Emission-BestValue.csv', index=False)
    pd.concat([pd.Series(np.arange(374)+1, name = 'AreaID'), 
               pd.DataFrame(np.vstack((aveEER_UKareaID_BestVal, 
                                       aveElecUse_UKareaID_BestVal, 
                                       aveGasUse_UKareaID_BestVal, 
                                       aveEmissions_UKareaID_BestVal)).T, 
                            columns = np.array(['Max EER', 'Min Elec', 'Min Gas', 'Min Emissions']))], axis=1).to_csv(
                    '../Visualization_Data/MiddlewareGeneratedOutputs/UK/UK-BestValue.csv', index=False)
    
    
    # pd.concat([pd.Series(np.arange(282)+1, name = 'AreaID'), 
    #            pd.Series(aveEER_ManAreaID_BestVal, name = 'Best Value (Max for EER; Min for Elec/Gas/Emissions)')], axis=1).to_csv(
    #                 '../Visualization_Data/MiddlewareGeneratedOutputs/Manchester/EER/Manchester-EER-BestValue.csv', index=False)
    # pd.concat([pd.Series(np.arange(282)+1, name = 'AreaID'), 
    #            pd.Series(aveElecUse_ManAreaID_BestVal, name = 'Best Value (Max for EER; Min for Elec/Gas/Emissions)')], axis=1).to_csv(
    #                 '../Visualization_Data/MiddlewareGeneratedOutputs/Manchester/Elec/Manchester-Elec-BestValue.csv', index=False)
    # pd.concat([pd.Series(np.arange(282)+1, name = 'AreaID'), 
    #            pd.Series(aveGasUse_ManAreaID_BestVal, name = 'Best Value (Max for EER; Min for Elec/Gas/Emissions)')], axis=1).to_csv(
    #                 '../Visualization_Data/MiddlewareGeneratedOutputs/Manchester/Gas/Manchester-Gas-BestValue.csv', index=False)
    # pd.concat([pd.Series(np.arange(282)+1, name = 'AreaID'), 
    #            pd.Series(aveEmissions_ManAreaID_BestVal, name = 'Best Value (Max for EER; Min for Elec/Gas/Emissions)')], axis=1).to_csv(
    #                 '../Visualization_Data/MiddlewareGeneratedOutputs/Manchester/Emission/Manchester-Emission-BestValue.csv', index=False)
    pd.concat([pd.Series(np.arange(282)+1, name = 'AreaID'), 
               pd.DataFrame(np.vstack((aveEER_ManAreaID_BestVal, 
                                       aveElecUse_ManAreaID_BestVal, 
                                       aveGasUse_ManAreaID_BestVal, 
                                       aveEmissions_ManAreaID_BestVal)).T, 
                            columns = np.array(['Max EER', 'Min Elec', 'Min Gas', 'Min Emissions']))], axis=1).to_csv(
                    '../Visualization_Data/MiddlewareGeneratedOutputs/Manchester/Manchester-BestValue.csv', index=False)
    
    
    pd.concat([pd.Series(np.arange(39)+1, name = 'BuildingID'), 
               pd.DataFrame(np.vstack((local39BldgEER_BestVal, 
                                       local39BldgElecUse_BestVal, 
                                       local39BldgGasUse_BestVal, 
                                       local39BldgEmissions_BestVal)).T, 
                            columns = np.array(['Max EER', 'Min Elec', 'Min Gas', 'Min Emissions']))], axis=1).to_csv(
                    '../Visualization_Data/MiddlewareGeneratedOutputs/LocalView/Metrics/LocalView-BestValue.csv', index=False)

