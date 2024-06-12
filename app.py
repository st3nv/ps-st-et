import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind
from stoc import stoc
import zipfile
import os
import shutil
import statsmodels.api as sm 
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.stats.multicomp import MultiComparison
from itertools import combinations
import scipy.stats as stats

import warnings
warnings.filterwarnings("ignore")

color_p= ["#1984c5", "#22a7f0", "#63bff0", "#a7d5ed", "#e2e2e2", "#e1a692", "#de6e56", "#e14b31", "#c23728"]

# Function definitions
shared_columns = ['idx','dimension', 'rot_type', 'angle', 'mirror', 'wm', 
                  'pair_id', 'obj_id', 'orientation1', 'orientation2', 'image_path_1', 'image_path_2',
                  'marker_id', 'correctAns', 'vivid_response', 'key_resp_vivid_slider_control.keys', 'key_resp_vivid_slider_control.rt', 'participant', 'condition_file']

def get_ans_key(row):
    keys_possible_cols = ['key_resp.keys', 'key_resp_3.keys', 'key_resp_6.keys']
    rt_possible_cols = ['key_resp.rt', 'key_resp_3.rt', 'key_resp_6.rt']
    for key, rt in zip(keys_possible_cols, rt_possible_cols):
        if not pd.isna(row[key]) and row[key] != '':
            return row[key], row[rt]
    return np.nan, np.nan

def get_strategy_response(row):
    if (not pd.isna(row['key_resp_strat_control.keys'])) and (row['key_resp_strat_control.keys'] != 'None') and (row['key_resp_strat_control.keys'] != ''):
        try:    
            strat_resp_list = eval(row['key_resp_strat_control.keys'])
            if len(strat_resp_list) > 0:
                last_key = strat_resp_list[-1]
                if last_key == 'rshift':
                    return 4
                elif last_key == 'slash':
                    return 3
                elif last_key == 'period':
                    return 2
                elif last_key == 'comma':
                    return 1
        except:
            print(row['key_resp_strat_control.keys'])
    return np.nan

def get_vivid_response(row):
    if (not pd.isna(row['key_resp_vivid_slider_control.keys'])) and (row['key_resp_vivid_slider_control.keys'] != 'None') and (row['key_resp_vivid_slider_control.keys'] != ''):
        try:    
            vivid_resp_list = eval(row['key_resp_vivid_slider_control.keys'])
            if len(vivid_resp_list) > 0:
                last_key = vivid_resp_list[-1]
                if last_key == 'rshift':
                    return 4
                elif last_key == 'slash':
                    return 3
                elif last_key == 'period':
                    return 2
                elif last_key == 'comma':
                    return 1
        except:
            print(row['key_resp_vivid_slider_control.keys'])
    return np.nan

def get_block(row):
    if row['dimension'] == '2D':
        if row['wm'] == False:
            return '2D_single'
        elif row['wm'] == True:
            return '2D_wm'
        
    elif row['dimension'] == '3D':
        if row['rot_type'] == 'p':
            if row['wm'] == False:
                return '3Dp_single'
            elif row['wm'] == True:
                return '3Dp_wm'
        elif row['rot_type'] == 'd':
            if row['wm'] == False:
                return '3Dd_single'
            elif row['wm'] == True:
                return '3Dd_wm'

def get_corr(row):
    if row['ans_key'] is np.nan:
        return np.nan
    else:
        if row['correctAns'] == row['ans_key']:
            return 1
        else:
            return 0


def parse_excel(df):
    df_blocks = df[~df['dimension'].isna()]
    df_strat = df[~df['key_resp_strat_control.keys'].isna()]
    df_strat = df_strat[['condition_file', 'key_resp_strat_control.keys', 'key_resp_strat_control.rt']]
    df_blocks.reset_index(drop=True, inplace=True)
    df_blocks['idx'] = df_blocks.index
    df_parsed = pd.DataFrame(columns=shared_columns)
    df_parsed['ans_key'] = np.nan
    df_parsed['rt'] = np.nan
    # iterate over the rows of the dataframe to get the ans keys, corr, rt by get_ans_key function
    for idx, row in df_blocks.iterrows():
        key, rt = get_ans_key(row)
        df_parsed.loc[idx, 'ans_key'] = key
        df_parsed.loc[idx, 'rt'] = rt
        for col in shared_columns:
            df_parsed.loc[idx, col] = row[col]
            
        # replace all 'None' values with np.nan
    df_parsed.replace('None', np.nan, inplace=True)
    df_parsed['vivid_response'] = df_parsed.apply(get_vivid_response, axis=1)

    # fill na values in 'rot_type', 'pair_id', 'orientation1', 'orientation2', 'image_path_2' with not applicable
    for col in ['rot_type', 'pair_id', 'orientation1', 'orientation2', 'image_path_2']:
        df_parsed[col].fillna('na', inplace=True)
        
    df_parsed['block'] = df_parsed.apply(get_block, axis=1)
    df_parsed['corr'] = df_parsed.apply(get_corr, axis=1)
    
    df_parsed = df_parsed.merge(df_strat, on='condition_file', how='left')
    df_parsed['strategy_response'] = df_parsed.apply(get_strategy_response, axis=1)
    
    df_parsed['mini_block'] = df_parsed['condition_file'].apply(lambda x: x.split('/')[1].split('.')[0]) 
    df_parsed.drop(columns=['condition_file'], inplace=True)
    return df_parsed



# make a new folder 'temp' to store the unzipped files and empty it if it already exists
import os
if os.path.exists('temp'):
    shutil.rmtree('temp')
os.makedirs('temp')

# Streamlit app
st.set_page_config(layout="wide")
st.title("Problem solving Eye Tracking Analysis (May 30 version)")

uploaded_behavior_file = st.file_uploader("Upload the zipped file of behavorial data", type="zip")
uploaded_et_file = st.file_uploader("Upload the zipped file of eye tracking data", type="zip")

if uploaded_behavior_file and uploaded_et_file:
    toc = stoc()
    
    # Unzip behavior file
    with zipfile.ZipFile(uploaded_behavior_file, "r") as z:
        z.extractall("behavior")
    unzipped_behavior_files = os.listdir("behavior")
    df_bh_parsed = pd.DataFrame()
    success_parsed_participant = []
    for file in unzipped_behavior_files:
        if file.endswith('.csv'):
            try:
                df = pd.read_csv(f"behavior/{file}")
                df_parsed = parse_excel(df)
                df_bh_parsed = pd.concat([df_bh_parsed, df_parsed], axis=0)
                success_parsed_participant.append(str(df_parsed['participant'].unique()[0]))
            except Exception as e:
                st.write(f"> Error parsing {file}: {e}")
    df_bh_parsed.reset_index(drop=True, inplace=True)
    success_parsed_participant = sorted(success_parsed_participant)
    st.write(f"Successfully parsed bahavioral data of participants: {success_parsed_participant}. ", "Total number of participants: ", len(success_parsed_participant))

    # Unzip and parse eye tracking data
    with zipfile.ZipFile(uploaded_et_file, "r") as z:
        z.extractall("et")
    
    unzipped_et_files = os.listdir("et")
    success_parsed_participant_et = []
    df_et_parsed = pd.DataFrame()
    for file in unzipped_et_files:
        if file.endswith('.csv'):
            try:
                participant_id = str(int(file.split('_')[0]))
                df_et = pd.read_csv(os.path.join("et", file))
                df_et['participant'] = participant_id
                df_et['event'] = df_et['event'].astype(str)
                df_et = df_et[df_et['event'] != '-1']
                df_et['marker_id'] = df_et['event'].apply(lambda x: x.split('_')[0])
                df_et['event_name'] = df_et['event'].apply(lambda x: x.split('_')[1])
                df_et.drop(columns=['event'], inplace=True)
                df_et_parsed = pd.concat([df_et_parsed, df_et], axis=0)
                success_parsed_participant_et.append(participant_id)
            except Exception as e:
                st.write(f"> Error parsing {file}: {e}")
                
    df_et_parsed.reset_index(drop=True, inplace=True)
    df_et_parsed['participant'] = df_et_parsed['participant'].astype(int)
    df_et_parsed['marker_id'] = df_et_parsed['marker_id'].astype(int)
    df_bh_parsed['marker_id'] = df_bh_parsed['marker_id'].astype(int)
    df_bh_parsed['participant'] = df_bh_parsed['participant'].astype(int)
        
    success_parsed_participant_et = sorted(success_parsed_participant_et)
    st.write(f"Successfully parsed eye tracking data of participants: {success_parsed_participant_et}. ", "Total number of participants: ", len(success_parsed_participant_et))
    
    # check if the number of participants in behavior and eye tracking data matches
    if set(success_parsed_participant) == set(success_parsed_participant_et):
        st.write("Number of participants in behavior and eye tracking data matches.")
    else:
        st.write("Number of participants in behavior and eye tracking data does not match.")

    st.write("Parsed behavioral data:")
    st.dataframe(df_bh_parsed.head())
    # st.download_button(label="Download parsed behavioral data", data=df_bh_parsed.to_csv(), file_name="parsed_behavioral_data.csv", mime="text/csv")
        
    st.write("Parsed eye tracking data:")
    st.dataframe(df_et_parsed.head())
    # st.download_button(label="Download parsed eye tracking data", data=df_et_parsed.to_csv(), file_name="parsed_eye_tracking_data.csv", mime="text/csv")
    
    # combining eyetracking and behavioral data
    # et left join bh
    st.write("Merging eye tracking and behavioral data...")
    df_combined = df_et_parsed.merge(df_bh_parsed, on=['participant', 'marker_id'], how='left')
    if df_combined['mini_block'].isna().sum() > 0:
        st.write(f"Participants with missing rows when merging: {df_combined[df_combined['mini_block'].isna()]['participant'].unique()}")
    else:
        st.write("Successfully merged eye tracking and behavioral data.")
    st.write("Combined data:")
    st.dataframe(df_combined.head())
    # st.download_button(label="Download combined data", data=df_combined.to_csv(), file_name="combined_data.csv", mime="text/csv")
    
    # missing rows
    if df_combined['mini_block'].isna().sum() > 0:
        st.write("Missing rows in combined data:")
        missing_rows = df_combined[df_combined['mini_block'].isna()]
        st.dataframe(missing_rows)
    
    # delete participants selectbox
    delete_participants = st.multiselect("Delete participants (None by default)", success_parsed_participant)
    if delete_participants:
        df_bh_parsed = df_bh_parsed[~df_bh_parsed['participant'].isin(delete_participants)]
        df_et_parsed = df_et_parsed[~df_et_parsed['participant'].isin(delete_participants)]
        df_combined = df_combined[~df_combined['participant'].isin(delete_participants)]
        st.write("Successfully deleted participants: ", delete_participants)
    
    # Use only correct responses
    use_correct_responses = st.sidebar.checkbox("Use only correct responses", value=False)
    if use_correct_responses:
        df_combined = df_combined[df_combined['corr'] == 1]
    
    event_list = df_combined['event_name'].unique()
    event_list.sort()
    event_interested = st.sidebar.multiselect("Select periods to analyze", event_list, ['trial', 'img1', 'img2'])
    st.sidebar.write("P.S. trial: Image in single trial; img1, img2: Images in wm; que, que1, que2: Que words; vivid: Vividness rating")
    
    df_analyze = df_combined[df_combined['event_name'].isin(event_interested)]
    
    toc.h2("Pupil Size Analysis")
    df_analyze['pupil_size'] = (df_analyze['left_pupil_diameter'] + df_analyze['right_pupil_diameter']) / 2
    # by participant
    toc.h3('By block')
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        df_pupil_block = df_analyze.groupby(['block']).agg({'pupil_size': ['mean', 'std']}).reset_index()
        df_pupil_block.columns = ['block', 'mean', 'std']
        st.dataframe(df_pupil_block)
    with col2:
        fig, ax = plt.subplots(figsize=(6, 4), dpi=200)
        # sort the df by block
        df_analyze_block_sorted = df_analyze.sort_values('block')
        sns.barplot(x='block', y='pupil_size', data=df_analyze_block_sorted, palette=color_p, ax=ax, capsize=0.1)
        ax.set_xlabel('Block', fontsize=14)
        ax.set_ylabel('Pupil size', fontsize=14)
        plt.title('Pupil size by block', fontsize=16)
        sns.despine()
        st.pyplot(fig)
    with col3:
        df_pupil_block_participant = df_analyze.groupby(['participant', 'block'])['pupil_size'].mean().reset_index()
        fig, ax = plt.subplots(figsize=(6, 5), dpi=200)
        sns.barplot(data=df_pupil_block_participant, x='block', y='pupil_size', hue='participant', ax=ax, palette=color_p)
        sns.despine()
        plt.legend(loc='upper right', bbox_to_anchor=(1.2, 1))
        st.pyplot(fig)
    
    # By wm
    toc.h3('By Single/WM')
    col1, col2, col3 = st.columns(3)
    df_pupil_analyze_wm = df_analyze.copy()
    df_pupil_analyze_wm['wm'] = df_pupil_analyze_wm['wm'].apply(lambda x: 'WM' if x else 'Single')
    with col1:
        df_pupil_wm = df_pupil_analyze_wm.groupby(['wm']).agg({'pupil_size': ['mean', 'std']}).reset_index()
        df_pupil_wm.columns = ['wm', 'mean', 'std']
        st.dataframe(df_pupil_wm)
    with col2:
        fig, ax = plt.subplots(figsize=(6, 4), dpi=200)
        sns.barplot(x='wm', y='pupil_size', data=df_pupil_analyze_wm, palette=color_p, ax=ax, capsize=0.1)
        ax.set_xlabel('Single/WM', fontsize=14)
        ax.set_ylabel('Pupil size', fontsize=14)
        plt.title('Pupil size by Single/WM', fontsize=16)
        sns.despine()
        st.pyplot(fig)
    with col3:
        df_pupil_wm_participant = df_pupil_analyze_wm.groupby(['participant', 'wm'])['pupil_size'].mean().reset_index()
        fig, ax = plt.subplots(figsize=(6, 5), dpi=200)
        sns.barplot(data=df_pupil_wm_participant, x='wm', y='pupil_size', hue='participant', ax=ax, palette=color_p)
        sns.despine()
        plt.legend(loc='upper right', bbox_to_anchor=(1.2, 1))
        plt.title('Pupil size by Single/WM for each participant', fontsize=14)
        st.pyplot(fig)
        
    # by angle
    toc.h3('By angle')
    col1, col2, col3 = st.columns(3)
    with col1:
        df_pupil_angle = df_analyze.groupby(['angle']).agg({'pupil_size': ['mean', 'std']}).reset_index()
        df_pupil_angle.columns = ['angle', 'mean', 'std']
        st.dataframe(df_pupil_angle)
    with col2:
        fig, ax = plt.subplots(figsize=(6, 4), dpi=200)
        sns.barplot(x='angle', y='pupil_size', data=df_analyze, palette=color_p, ax=ax, capsize=0.1)
        ax.set_xlabel('Angle', fontsize=14)
        ax.set_ylabel('Pupil size', fontsize=14)
        plt.title('Pupil size by angle', fontsize=14)
        sns.despine()
        st.pyplot(fig)
    with col3:
        df_pupil_angle_participant = df_analyze.groupby(['participant', 'angle'])['pupil_size'].mean().reset_index()
        fig, ax = plt.subplots(figsize=(6, 5), dpi=200)
        sns.barplot(data=df_pupil_angle_participant, x='angle', y='pupil_size', hue='participant', ax=ax, palette=color_p)
        sns.despine()
        plt.legend(loc='upper right', bbox_to_anchor=(1.2, 1))
        plt.title('Pupil size by angle for each participant', fontsize=14)
        st.pyplot(fig)
        
    # By dimension
    toc.h3('By dimension')
    col1, col2, col3 = st.columns(3)
    with col1:
        df_pupil_dimension = df_analyze.groupby(['dimension']).agg({'pupil_size': ['mean', 'std']}).reset_index()
        df_pupil_dimension.columns = ['dimension', 'mean', 'std']
        st.dataframe(df_pupil_dimension)
    with col2:
        fig, ax = plt.subplots(figsize=(6, 4), dpi=200)
        sns.barplot(x='dimension', y='pupil_size', data=df_analyze, palette=color_p, ax=ax, capsize=0.1)
        ax.set_xlabel('Dimension', fontsize=14)
        ax.set_ylabel('Pupil size', fontsize=14)
        plt.title('Pupil size by dimension', fontsize=14)
        sns.despine()
        st.pyplot(fig)
    with col3:
        df_pupil_dimension_participant = df_analyze.groupby(['participant', 'dimension'])['pupil_size'].mean().reset_index()
        fig, ax = plt.subplots(figsize=(6, 5), dpi=200)
        sns.barplot(data=df_pupil_dimension_participant, x='dimension', y='pupil_size', hue='participant', ax=ax, palette=color_p)
        sns.despine()
        plt.legend(loc='upper right', bbox_to_anchor=(1.2, 1))
        plt.title('Pupil size by dimension for each participant', fontsize=14)
        st.pyplot(fig)
    
    # By event
    toc.h3('By event')
    col1, col2, col3 = st.columns(3)
    with col1:
        df_pupil_event = df_analyze.groupby(['event_name']).agg({'pupil_size': ['mean', 'std']}).reset_index()
        df_pupil_event.columns = ['event_name', 'mean', 'std']
        st.dataframe(df_pupil_event)
    with col2:
        fig, ax = plt.subplots(figsize=(6, 4), dpi=200)
        sns.barplot(x='event_name', y='pupil_size', data=df_analyze, palette=color_p, ax=ax, capsize=0.1)
        ax.set_xlabel('Event', fontsize=14)
        ax.set_ylabel('Pupil size', fontsize=14)
        plt.title('Pupil size by event', fontsize=14)
        sns.despine()
        st.pyplot(fig)
    with col3:
        df_pupil_event_participant = df_analyze.groupby(['participant', 'event_name'])['pupil_size'].mean().reset_index()
        fig, ax = plt.subplots(figsize=(6, 5), dpi=200)
        sns.barplot(data=df_pupil_event_participant, x='event_name', y='pupil_size', hue='participant', ax=ax, palette=color_p)
        sns.despine()
        plt.legend(loc='upper right', bbox_to_anchor=(1.2, 1))
        plt.title('Pupil size by event for each participant', fontsize=14)
        st.pyplot(fig)
    
    # Pupil size over time
    toc.h3('Pupil size over time')
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        # avg pupil size over trial idx for each participant
        df_pupil_over_time = df_analyze.groupby(['participant', 'idx']).agg({'pupil_size': 'mean'}).reset_index()
        fig, ax = plt.subplots(figsize=(8, 5), dpi=200)
        sns.lineplot(data=df_pupil_over_time, x='idx', y='pupil_size', hue='participant', ax=ax, palette=color_p)
        sns.despine()
        plt.title('Pupil size in each trial', fontsize=14)
        plt.legend(loc='upper right', bbox_to_anchor=(1.15, 1))
        st.pyplot(fig)
    
    # ANOVA
    toc.h3('ANOVA')

    df_analyze_for_anova = df_analyze.copy()[['pupil_size', 'participant', 'angle', 'dimension', 'block', 'wm', 'event_name']]
    
    # anova multi-select
    anova_factors = st.multiselect("Select variables for ANOVA", ['wm', 'dimension', 'angle', 'block', 'event_name'], key = 'anova_factors', default= ['event_name', 'dimension'])
    
    # Generate all possible combinations of variables
    combs = sum([list(map(list, combinations(anova_factors, i))) for i in range(1, len(anova_factors) + 1)], [])

    # Generate the formula string
    formula = 'pupil_size ~ ' + ' + '.join(['C(' + '):C('.join(c) + ')' for c in combs])
    anova_pupil = ols(formula, data=df_analyze_for_anova).fit()
    anova_table = sm.stats.anova_lm(anova_pupil, typ=2)
    st.write(anova_table)
    
    st.write("Post-hoc test:")
    factors = st.multiselect("Select factors for post-hoc test", anova_factors, key = 'factors', default= anova_factors)
    # multi compare
    tmp = df_analyze_for_anova[factors]
    tmp['group_label'] = tmp.apply(lambda x: '_'.join(x), axis=1)
    df_analyze_for_anova['group_label'] = tmp['group_label']
    df_analyze_for_anova.dropna(subset=['pupil_size'], inplace=True)
    mc = MultiComparison(df_analyze_for_anova['pupil_size'], df_analyze_for_anova['group_label'])
    mc_results = mc.tukeyhsd()
    st.write(mc_results)
    
    toc.h2("Gaze Point Analysis")
    
    # gaze heatmap for each participant
    # gaze_point_x and gaze_point_y are in 0-1 scale
    toc.h3("2D Gaze heatmap")
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    
    participant_list = df_analyze['participant'].unique()
    # sort but 10 is after 5
    participant_list = sorted(participant_list, key=lambda x: int(x))
    for participant in participant_list:
        df_participant = df_analyze[df_analyze['participant'] == participant]
        with col1:
            # during 2d single trials
            df_participant_2d_single = df_participant[(df_participant['block'] == '2D_single') & (df_participant['event_name'] == 'trial')]
            fig, ax = plt.subplots(figsize=(4, 3), dpi=200)
            sns.kdeplot(data=df_participant_2d_single, x='gaze_point_x', y='gaze_point_y', fill=True, cmap='viridis', ax=ax)
            plt.xlim(0, 1)
            plt.ylim(0, 1)
            ax.set_title(f'Participant {participant} 2D Single')
            st.pyplot(fig)
        with col2:
            # during 2d wm trials
            df_participant_2d_wm = df_participant[(df_participant['block'] == '2D_wm') & (df_participant['event_name'] == 'img1')]
            fig, ax = plt.subplots(figsize=(4, 3), dpi=200)
            sns.kdeplot(data=df_participant_2d_wm, x='gaze_point_x', y='gaze_point_y', fill=True, cmap='viridis', ax=ax)
            plt.xlim(0, 1)
            plt.ylim(0, 1)
            ax.set_title(f'Participant {participant} 2D WM - img1')
            st.pyplot(fig)
        with col3:
            # during 2D_wm second image
            df_participant_3d_single = df_participant[(df_participant['block'] == '2D_wm') & (df_participant['event_name'] == 'img2')]
            fig, ax = plt.subplots(figsize=(4, 3), dpi=200)
            sns.kdeplot(data=df_participant_3d_single, x='gaze_point_x', y='gaze_point_y', fill=True, cmap='viridis', ax=ax)
            plt.xlim(0, 1)
            plt.ylim(0, 1)
            ax.set_title(f'Participant {participant} 2D WM - img2')
            st.pyplot(fig)
        with col4:
            # during 3d single trials
            df_participant_3d_single = df_participant[(df_participant['block'] == '3Dp_single') & (df_participant['event_name'] == 'trial')]
            fig, ax = plt.subplots(figsize=(4, 3), dpi=200)
            sns.kdeplot(data=df_participant_3d_single, x='gaze_point_x', y='gaze_point_y', fill=True, cmap='viridis', ax=ax)
            plt.xlim(0, 1)
            plt.ylim(0, 1)
            ax.set_title(f'Participant {participant} 3D Single')
            st.pyplot(fig)
        with col5:
            # during 3d wm trials
            df_participant_3d_wm = df_participant[(df_participant['block'] == '3Dp_wm') & (df_participant['event_name'] == 'img1')]
            fig, ax = plt.subplots(figsize=(4, 3), dpi=200)
            sns.kdeplot(data=df_participant_3d_wm, x='gaze_point_x', y='gaze_point_y', fill=True, cmap='viridis', ax=ax)
            plt.xlim(0, 1)
            plt.ylim(0, 1)
            ax.set_title(f'Participant {participant} 3D WM - img1')
            st.pyplot(fig)
        with col6:
            # during 3D_wm second image
            df_participant_3d_single = df_participant[(df_participant['block'] == '3Dp_wm') & (df_participant['event_name'] == 'img2')]
            fig, ax = plt.subplots(figsize=(4, 3), dpi=200)
            sns.kdeplot(data=df_participant_3d_single, x='gaze_point_x', y='gaze_point_y', fill=True, cmap='viridis', ax=ax)
            plt.xlim(0, 1)
            plt.ylim(0, 1)
            ax.set_title(f'Participant {participant} 3D WM - img2')
            st.pyplot(fig)
            
            
            
            
    
    toc.toc()