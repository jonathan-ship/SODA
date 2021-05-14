import sys
import math
import scipy
import scipy.misc

import numpy as np
import pandas as pd
import docplex.cp.utils_visu as visu
import matplotlib.pyplot as plt

from pylab import rcParams
from collections import namedtuple
from collections import OrderedDict
from docplex.cp.model import *


# 각 팀별 할당된 세부 액티비티에 대한 정보를 불러오는 함수
def import_data_by_team(filepath, team):
    # 중일정 계획에 대한 데이터
    df_activity_all = pd.read_excel(filepath, sheet_name=0, usecols=['호선', '블록', '작업팀', '중일정착수(계획)', '중일정완료(계획)'])
    df_activity = df_activity_all[df_activity_all['작업팀'].isin(team)]
    df_activity = df_activity.sort_values(by=['호선', '블록'], ascending=True)
    df_activity['중일정완료(계획)'] = pd.to_datetime(df_activity['중일정완료(계획)'], format='%Y%m%d')
    df_activity['중일정착수(계획)'] = pd.to_datetime(df_activity['중일정착수(계획)'], format='%Y%m%d')

    min_day = df_activity['중일정착수(계획)'].min()
    max_day = df_activity['중일정완료(계획)'].max()
    df_activity['중일정완료(계획)'] = (df_activity['중일정완료(계획)'] - min_day).dt.days * 9 + 8
    df_activity['중일정착수(계획)'] = (df_activity['중일정착수(계획)'] - min_day).dt.days * 9
    df_activity.dropna(subset=['블록'], inplace=True)
    activity = df_activity.values.tolist()

    # 세부 액티비티에 대한 데이터
    df_act_all = pd.read_excel(filepath, sheet_name=1,
                               usecols=['호선', '블록', '작업팀', '필요인원', '작업시간', '공종', 'ID', '선행Act1', '관계종류1',
                                        '선행Act2', '관계종류2', 'LAG'])
    df_act_team = df_act_all[df_act_all['작업팀'].isin(team)]
    df_act_exception = df_act_team[df_act_team['호선'].isin(df_activity['호선'])]
    df_act = df_act_exception[df_act_exception['블록'].isin(df_activity['블록'])]
    df_act = df_act.sort_values(by=['호선', '블록'], ascending=True)
    df_act.dropna(subset=['블록'], inplace=True)
    df_act = df_act.reset_index(drop=True)

    acts = OrderedDict()
    for i, act in df_act.iterrows():
        # print(str(act['호선'])+str(act['블록'])+str(act['ID']))
        acts[str(act['호선']) + str(act['블록']) + str(act['ID'])] \
            = Act(work_id=str(act['호선']) + str(act['블록']) + str(act['ID']),
                  project=act['호선'],
                  block=act['블록'],
                  team=act['작업팀'],
                  worker=act['필요인원'],
                  lead_time=act['작업시간'],
                  process=act['공종'],
                  id=act['ID'],
                  precedence1=act['선행Act1'],
                  relation1=act['관계종류1'],
                  precedence2=act['선행Act2'],
                  relation2=act['관계종류2'],
                  project_block=str(act['호선']) + str(act['블록']),
                  lag=act['LAG'])

    # 작업팀에 대한 데이터
    df_worker = pd.read_excel(filepath, sheet_name=2, header=1)
    worker = df_worker[team[0]].values.tolist()

    # 휴일 정보에 대한 데이터
    df_calender = pd.read_excel(filepath, sheet_name=3, usecols=['일자', '휴일구분'])
    df_calender['일자'] = pd.to_datetime(df_calender['일자'], format='%Y%m%d')
    df_calender = df_calender[(df_calender['일자'] >= min_day) & (df_calender['일자'] <= max_day)]
    df_calender = df_calender.sort_values(by=['일자'], ascending=True)
    df_calender = df_calender[df_calender['휴일구분'].isin([3])]
    df_calender['일자'] = (df_calender['일자'] - min_day).dt.days * 9
    calender = df_calender['일자'].values.tolist()

    return activity, acts, worker, calender


# 세부 액티비티에 대한 데이터를 담은 클래스 정의
class Act:
    def __init__(self, work_id=None, project=None, block=None, team=None, worker=None, lead_time=1, process=None, id=None,
                 precedence1=None, relation1=None, precedence2=None, relation2=None, project_block=None,lag=None):
        self.work_id = work_id  # "호선 + 블록 + 세부 액티비티 ID"
        self.project = project  # 호선 번호
        self.block = block  # 블록 번호
        self.team = team  # 작업팀
        self.worker = worker  # 필요 인원
        self.lead_time = lead_time  # 필요 기간
        self.process = process  # 공종
        self.id = id  # 세부 액티비티 ID
        self.precedence1 = precedence1  # 첫 번째 연결관계에 대응하는 세부 액티비티 ID
        self.relation1 = relation1  # 첫 번째 연결관계 종류
        self.precedence2 = precedence2  # 두 번째 연결관계에 대응하는 세부 액티비티 ID
        self.relation2 = relation2  # 두 번째 연결관계 종류
        self.project_block = project_block  # 호선 번호 + 블록 번호
        self.lag = lag  # 연결관계에서의 래그


if __name__ == '__main__':
    # 파일 경로 설정
    filepath = './data/(전달)서울대학교_시뮬레이션_데이터_201201_201231.xlsx'

    # 작업팀 설정
    team = ['T9']

    # 설정된 작업팀에 할당된 세부 액티비티 데이터 로딩
    activity, acts, worker, calender = import_data_by_team(filepath, team)

    # CSP 모델 생성
    mdl = CpoModel()

    # 휴일 설정
    Breaks = []
    for i in range(len(calender)):
        Breaks.append((calender[i], calender[i] + 9))
    print(Breaks)

    Break = namedtuple('Break', ['start', 'end'])
    Calendars = {}
    mymax = max(v for k, v in Breaks)
    step = CpoStepFunction()
    step.set_value(0, mymax+100000000, 100)
    for b in Breaks:
        t = Break(*b)
        step.set_value(t.start, t.end, 0)
    Calendars = step

    # 각 세부 액티비티에 대한 interval variable 생성
    pj_bl_list = []
    for i in range(len(activity)):
        pj_bl_list.append(str(activity[i][0]) + str(activity[i][1]))
    print(pj_bl_list)

    act_var = dict()
    for i in range(len(pj_bl_list)):
        for key, value in acts.items():
            if str(acts[key].project_block) == str(pj_bl_list[i]):
                act_var[key] = mdl.interval_var(start=(activity[i][3],100000000), size=value.lead_time,
                                                intensity=Calendars, name=value.work_id+"_"+str(value.process))

    # 세부 액티비티 간의 선후행 연결관계에 대한 제약조건 설정
    for key, value in acts.items():
        if acts[key].relation1 == "FS" and acts[key].precedence1:
            mdl.add(mdl.end_before_start(act_var[str(acts[key].project_block) + str(acts[key].precedence1)],
                                         act_var[key], delay=acts[key].lag))
        if acts[key].relation2 == "FS" and acts[key].precedence2:
            mdl.add(mdl.end_before_start(act_var[str(acts[key].project_block) + str(acts[key].precedence2)],
                                         act_var[key]))

    # 작업팀의 공종 별 작업 인력에 대한 제약 조건 설정
    worker_loads_process = [0 for i in range(len(worker))]
    for i in range(len(worker)):
        worker_loads_process[i] = step_at(0, 0)
        for key, value in act_var.items():
            if acts[key].process == "W" + str(i + 1):
                worker_loads_process[i] += mdl.pulse(value, acts[key].worker)
        mdl.add(worker_loads_process[i] <= worker[i])

    # 첫 번째 목적함수(납기 지연 최소화) 설정
    # 지연 발생 시 일 단위로 패널티 부여
    pj_bl_var = dict()
    for i in range(len(pj_bl_list)):
        temp = []
        pj_bl_var[pj_bl_list[i]] = mdl.interval_var(name=pj_bl_list[i])
        for key, value in acts.items():
            if str(acts[key].project_block) == str(pj_bl_list[i]):
                temp.append(act_var[key])
        if len(temp) > 0:
            pj_bl_var[pj_bl_list[i]] = mdl.interval_var(name=pj_bl_list[i])
            mdl.add(mdl.span(pj_bl_var[pj_bl_list[i]], temp))

    penalty = 1
    obj1 = sum(max(mdl.end_of(pj_bl_var[pj_bl_list[i]]) - activity[i][4], 0) for i in range(len(pj_bl_list))) * penalty / 9

    # 두 번째 목적함수(전체 인력의 80%를 초과하는 기간 최소화) 설정
    # 전체 인력의 80%를 초과하도록 액티비티가 계획될 경우 패널티 부여
    standard = math.floor(sum(worker) * 0.8)

    use = dict()
    overuse = dict()
    for i in range(len(pj_bl_list)):
        for key, value in acts.items():
            if str(acts[key].project_block) == str(pj_bl_list[i]):
                use[key] = mdl.interval_var(start=(activity[i][3], 100000000), size=value.lead_time, intensity=Calendars,
                                            name="use" + value.work_id + "_" + str(value.process), optional=True)
                overuse[key] = mdl.interval_var(start=(activity[i][3], 100000000), size=value.lead_time, intensity=Calendars,
                                                name="overuse" + value.work_id + "_" + str(value.process), optional=True)
                mdl.add(mdl.alternative(act_var[key], [use[key], overuse[key]]))

    worker_loads_use = step_at(0, 0)
    for key, value in act_var.items():
        worker_loads_use += mdl.pulse(value, acts[key].worker)
    mdl.add(worker_loads_use <= standard)

    obj2 = mdl.sum(mdl.presence_of(overuse[key]) * value.lead_time for key, value in acts.items())

    # 두 목적함수의 가중합으로 최종 목적함수 설정
    w1 = 0.9
    w2 = 0.1

    obj = w1 * obj1 + w2 * obj2
    mdl.add(mdl.minimize(obj))

    # 해 탐색
    sol = mdl.solve(TimeLimit=300)

    # 탐색 결과 출력
    print("The objective value is " + str(sol.get_objective_values()[0]))
    print("delay: "+str(sum(max(sol.get_var_solution(pj_bl_var[pj_bl_list[i]]).get_end() - activity[i][4], 0)
                            for i in range(len(pj_bl_list))) * penalty / 9) + "days")

    last_end = 0

    for i in range(len(pj_bl_list)):
        print(pj_bl_list[i])
        print(sol.get_var_solution(pj_bl_var[pj_bl_list[i]]))

        for key, value in acts.items():
            if str(acts[key].project_block) == str(pj_bl_list[i]):
                print(sol.get_var_solution(act_var[key]))
                if last_end < sol.get_var_solution(act_var[key]).get_end():
                    last_end = sol.get_var_solution(act_var[key]).get_end()

    # 탐색 결과 가시화
    # 블록 별 스케줄링 결과
    rcParams['figure.figsize'] = 15, 60
    for i in range(len(pj_bl_list)):
        visu.panel(name=pj_bl_list[i])
        for key, value in acts.items():
            if str(acts[key].project_block) == str(pj_bl_list[i]):
                temp1 = sol.get_var_solution(act_var[key])
                visu.interval(temp1,'lightblue',value.id[4:])
    visu.show(pngfile="scheduling")

    # 전체 인력 사용 수준 그래프
    print("workforce level for" + str(team))
    rcParams['figure.figsize'] = 15, 3
    workforce=CpoStepFunction()
    workforce_standard=CpoStepFunction()
    workforce_whole=CpoStepFunction()
    for key, value in acts.items():
        wf=sol.get_var_solution(act_var[key])
        workforce.add_value(wf.get_start(),wf.get_end(), acts[key].worker)

    workforce_whole.add_value(0, last_end, sum(worker))
    workforce_standard.add_value(0,last_end ,standard)

    # 공종 별 인력 사용 수준 그래프
    visu.panel(name="Workers")
    visu.function(segments=workforce, style='area', origin=0, horizon=last_end, color='lightgreen')
    visu.function(segments=workforce_standard, style='line', origin=0, horizon=last_end)
    visu.function(segments=workforce_whole, style='line', origin=0, horizon=last_end, color='blue')
    visu.show(pngfile="overall_workforce")

    print(worker)
    for i in range(len(worker)):
        workforce_bywork = CpoStepFunction()
        workforce_limit = CpoStepFunction()
        visu.panel(name="Work " + str(i + 1))

        for key, value in acts.items():
            if acts[key].process == "W" + str(i + 1):
                itv = sol.get_var_solution(act_var[key])
                workforce_bywork.add_value(itv.get_start(), itv.get_end(), acts[key].worker)
        workforce_limit.add_value(0, last_end, worker[i])

        visu.function(segments=workforce_bywork, style='area', origin=0, horizon=last_end, color='lightblue')
        visu.function(segments=workforce_limit, style='line', origin=0, horizon=last_end)
        visu.show(pngfile="workforce for work " + str(i+1))