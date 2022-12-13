# custom_functions.py

import datetime
import ntpath
import random
import re
import sys
from itertools import chain, compress

import numpy as np
import pandas as pd

# import glob
# import socket
import scipy.optimize as opt
from sklearn.linear_model import LinearRegression, LogisticRegressionCV

from mouse_behavior_analysis_tools.utils.misc_utils import update_progress


def first_diff_zero(array):
    # define a function that returns only those indices of a binary!
    # vector (0 or 1)
    # where some values are first different than 0
    # create a new vector that is the same but shifted
    # move everything one space forward
    newarray = np.concatenate((0, array), axis=None)[0 : len(array)]
    difarray = array - newarray
    get_indexes = lambda x, xs: [
        i for (y, i) in zip(xs, range(len(xs))) if x == y
    ]
    # find which indexes are 1
    indexes = get_indexes(1, difarray)
    return indexes


def time_to_zero(input_list):
    return list(np.array(input_list) - input_list[0])


def ParseForTimes(files):
    # looks for 8digits followed by underscore and 6digits (bpod style)
    dates = []
    for title in files:
        try:
            match = re.search(r"\d{8}_\d{6}", ntpath.basename(title))
            dates.append(match.group())
        except Exception:
            dates.append("notFound")
    return dates


def BpodDatesToTime(inputDates):
    # assumes input style YYYYMMDD_HHMMSS
    # returns a time object
    outputDates = []
    for date in inputDates:
        try:
            x = datetime.datetime(
                int(date[0:4]),
                int(date[4:6]),
                int(date[6:8]),
                int(date[9:11]),
                int(date[11:13]),
            )
            outputDates.append(x)
        except Exception:
            outputDates.append("notFound")
    return outputDates


def PsychPerformance(trialsDif, sideSelected):
    # function to calculate psychometric performance and
    # fit logistic regression to the data
    # returns a dictionary

    if trialsDif.any():  # in case an empty thing is passed

        # masks to remove nans for logistic regression
        nan_mask = ~(np.isnan(trialsDif) | np.isnan(sideSelected))
        # logistic regression
        if len(np.unique(sideSelected)) > 1:
            clf = LogisticRegressionCV(cv=3).fit(
                trialsDif[nan_mask, np.newaxis], sideSelected[nan_mask]
            )
        else:
            # in case a model cannot be fitted
            # (e.g. mouse always goes to the left)
            # fit model on dummy data
            clf = LogisticRegressionCV(cv=3).fit(
                np.array([0, 0, 0, 100, 100, 100]).reshape(-1, 1),
                np.array([1, 0, 1, 0, 1, 0]),
            )
        # Calculate performance
        # Initialize values
        difficulty = np.unique(trialsDif[~np.isnan(trialsDif)])
        performance = np.full(len(difficulty), np.nan)
        for i in range(len(difficulty)):
            if np.nansum(sideSelected[trialsDif == difficulty[i]]) > 0:
                performance[i] = 100 * (
                    np.nanmean(sideSelected[trialsDif == difficulty[i]]) - 1
                )
            else:
                performance[i] = np.nan

        DictToReturn = {
            "Difficulty": difficulty,
            "Performance": performance,
            "Logit": clf,
        }
    else:
        DictToReturn = {}

    return DictToReturn


def splitOpto(SessionData):
    # SessionData comes from bpod: ExperimentData[x]['SessionData']
    # Returns two dictionaries

    Trials_normalMask = SessionData["OptoStim"] == 0
    Trials_optoMask = SessionData["OptoStim"] == 1

    # selection of normal and opto trials
    normalTrials_sideSelected = SessionData["FirstPoke"][Trials_normalMask]
    normalTrials_difficulty = SessionData["TrialHighPerc"][Trials_normalMask]
    optoTrials_sideSelected = SessionData["FirstPoke"][Trials_optoMask]
    optolTrials_difficulty = SessionData["TrialHighPerc"][Trials_optoMask]

    # create dictionaries
    NormalTrials = {
        "SideSelected": normalTrials_sideSelected,
        "Difficulty": normalTrials_difficulty,
    }

    OptoTrials = {
        "SideSelected": optoTrials_sideSelected,
        "Difficulty": optolTrials_difficulty,
    }

    return NormalTrials, OptoTrials


def generate_fake_data(trialsDif, sideSel):
    # Generates data for bootstrapping, sampling and replacing, so each
    # unique trialsDif maintains the same size

    fake_side_sel = np.empty_like(sideSel)
    for curr_diff in np.unique(trialsDif):
        diff_mask = trialsDif == curr_diff
        population = sideSel[diff_mask]
        fake_side_sel[diff_mask] = np.random.choice(
            population, len(population)
        )

    return fake_side_sel


def BootstrapPerformances(
    trialsDif, sideSelected, ntimes, prediction_difficulties
):
    # Bootstrap data and return logistic regression
    # predictions for each sampled model
    # remove nans
    nan_mask = ~(np.isnan(sideSelected) | np.isnan(trialsDif))
    difficulties = trialsDif[nan_mask]
    sideselection = sideSelected[nan_mask]

    predictPerFake = np.empty((len(prediction_difficulties), ntimes))
    for i in range(predictPerFake.shape[1]):
        # create fake data
        fake_data = generate_fake_data(difficulties, sideselection)
        try:
            clf_fake = LogisticRegressionCV(cv=3).fit(
                difficulties.reshape(-1, 1), fake_data
            )
            predictPerFake[:, i] = (
                100 * clf_fake.predict_proba(prediction_difficulties)[:, 1]
            )
        except Exception:
            # in case a model cannot be fitted
            # (e.g. mouse always goes to the left)
            # fit model on dummy data
            clf_fake = LogisticRegressionCV(cv=3).fit(
                np.array([0, 0, 0, 100, 100, 100]).reshape(-1, 1),
                np.array([1, 0, 1, 0, 1, 0]),
            )

    return predictPerFake


def SessionDataToDataFrame(
    AnimalID, ExperimentalGroup, SessionID, SessionData
):
    # function to create a dataframe out of the session
    # each trial is an entry on the dataframe

    # if the session is empty output a message
    if "nTrials" not in SessionData:
        print("Session is empty")
        return pd.DataFrame()

    numberOfTrials = SessionData["nTrials"]

    # protocol information
    ts = SessionData["TrialSettings"]
    protocols = [
        ts[0]["GUIMeta"]["TrainingLevel"]["String"][x]
        for x in [y["GUI"]["TrainingLevel"] - 1 for y in ts]
    ]
    stimulations = [
        ts[0]["GUIMeta"]["OptoStim"]["String"][x]
        for x in [y["GUI"]["OptoStim"] - 1 for y in ts]
    ]

    # muscimol
    yList = []
    for y in ts:
        try:
            yList.append(y["GUI"]["Muscimol"] - 1)
        except Exception:
            yList.append(0)
    muscimol = []
    for x in yList:
        try:
            muscimol.append(ts[0]["GUIMeta"]["Muscimol"]["String"][x])
        except Exception:
            muscimol.append("No")

    # punish method
    yList = []
    for y in ts:
        try:
            yList.append(y["GUI"]["Punish"] - 1)
        except Exception:
            yList.append(0)
    punish = []
    for x in yList:
        try:
            punish.append(ts[0]["GUIMeta"]["Punish"]["String"][x])
        except Exception:
            punish.append("No")

    # reward change
    yList = []
    reward_change_block = []
    for y in ts:
        try:
            yList.append(y["GUI"]["RewardChange"] - 1)
            reward_change_block.append(y["RewardChangeBlock"])
        except Exception:
            yList.append(0)
            reward_change_block.append(0)
    reward_change = []
    for x in yList:
        try:
            reward_change.append(ts[0]["GUIMeta"]["RewardChange"]["String"][x])
        except Exception:
            reward_change.append("No")

    if not np.logical_and(
        len(protocols) == numberOfTrials, len(stimulations) == numberOfTrials
    ):
        print(
            "protocols and/or stimulations length do\
            not match with the number of trials"
        )
        return pd.DataFrame()
    CenterPortDuration = [x["GUI"]["CenterPortDuration"] for x in ts]
    Contingency = [x["GUI"]["Contingency"] for x in ts]
    RewardAmount = [x["GUI"]["RewardAmount"] for x in ts]
    PunishDelay = [x["GUI"]["PunishDelay"] for x in ts]
    BiasCorrection = [x["GUI"]["BiasCorrection"] for x in ts]
    FullGUI = [x["GUI"] for x in ts]

    # trial events
    trev = [x["Events"] for x in SessionData["RawEvents"]["Trial"]]
    if not len(trev) == numberOfTrials:
        print("trial events length do not match with the number of trials")
        return pd.DataFrame()

    # trial states
    trst = [x["States"] for x in SessionData["RawEvents"]["Trial"]]
    if not len(trst) == numberOfTrials:
        print("trial states length do not match with the number of trials")
        return pd.DataFrame()

    # calculate the cumulative performance
    firstpokecorrect = SessionData["FirstPokeCorrect"][0:numberOfTrials]
    correct_cp = np.cumsum(firstpokecorrect == 1)
    incorrect_cp = np.cumsum(firstpokecorrect == 0)
    # the following line gives an error sometimes
    cumper = 100 * correct_cp / (correct_cp + incorrect_cp)

    # calculate when there is a side-switching event
    TriSide = np.array(SessionData["TrialSide"][0:numberOfTrials])
    SwitchSide = 1 * ((TriSide - np.insert(TriSide[:-1], 0, 0)) != 0)

    # add information about the choice in the previous trial'
    FirstPoke = SessionData["FirstPoke"][0:numberOfTrials]
    PrevTriChoice = np.insert(np.asfarray(FirstPoke[:-1]), 0, np.nan)

    # create a nice ID for the session (pretty date/time)
    prettyDate = SessionID.strftime("%b%d %H:%M")

    DFtoReturn = pd.DataFrame(
        {
            "AnimalID": pd.Series(np.repeat(AnimalID, numberOfTrials)).astype(
                "category"
            ),
            "ExperimentalGroup": pd.Series(
                np.repeat(ExperimentalGroup, numberOfTrials)
            ).astype("category"),
            "SessionTime": pd.Series(
                np.repeat(prettyDate, numberOfTrials)
            ).astype("category"),
            "FullSessionTime": np.repeat(SessionID, numberOfTrials),
            "Protocol": protocols,
            "Stimulation": stimulations,
            "Muscimol": muscimol,
            "RewardChange": reward_change,
            "RewardChangeBlock": reward_change_block,
            "CenterPortDuration": CenterPortDuration,
            "Contingency": Contingency,
            "RewardAmount": RewardAmount,
            "PunishDelay": PunishDelay,
            "Punish": punish,
            "BiasCorrection": BiasCorrection,
            "TrialIndex": list(range(numberOfTrials)),
            "TrialHighPerc": SessionData["TrialHighPerc"][0:numberOfTrials],
            "Outcomes": SessionData["Outcomes"][0:numberOfTrials],
            "OptoStim": SessionData["OptoStim"][0:numberOfTrials],
            "FirstPokeCorrect": firstpokecorrect,
            "FirstPoke": FirstPoke,
            "TrialSide": TriSide,
            "TrialSequence": SessionData["TrialSequence"][0:numberOfTrials],
            "ResponseTime": SessionData["ResponseTime"][0:numberOfTrials],
            "TrialStartTimestamp": SessionData["TrialStartTimestamp"],
            "CumulativePerformance": cumper,
            "SwitchSide": SwitchSide,
            "PreviousChoice": PrevTriChoice,
            "TrialEvents": trev,
            "TrialStates": trst,
            "FullGUI": FullGUI,
        }
    )

    return DFtoReturn


def identifyIdx(datatimes, ntrialsList, ntrials_thr):
    idxlist = []
    for i in range(len(datatimes)):
        if np.logical_or(
            datatimes[i] == "notFound", ntrialsList[i] < ntrials_thr
        ):
            idxlist.append(i)
    return sorted(idxlist, reverse=True)


# Analyze this with the optotrials as well
def AnalyzeSwitchTrials(df):
    # df is a dataframe containing the following columns:
    # 'SwitchSide'
    # 'FirstPokeCorrect'
    # 'SessionTime'
    # 'OptoStim'
    # it returns a different dataframe with information grouped for a bar plot

    # get info for the sessions
    sessionsID = pd.unique(df["SessionTime"])
    # initialize list to hold dataframes
    sessionsInfo = []

    # fill the new dataframe with info for each session
    for session in sessionsID:
        # get the dataframe for that session
        Sdf = df[df["SessionTime"] == session]
        # split the dataset into opto and normal
        Ndf = Sdf[Sdf["OptoStim"] == 0]
        Odf = Sdf[Sdf["OptoStim"] == 1]
        # percentage of correct trials on stay trials without stimulation
        StayNoStim = (
            100
            * np.sum(Ndf[Ndf["SwitchSide"] == 0]["FirstPokeCorrect"] == 1)
            / len(Ndf[Ndf["SwitchSide"] == 0])
        )
        # percentage of correct trials on switch trials without stimulation
        SwitchNoStim = (
            100
            * np.sum(Ndf[Ndf["SwitchSide"] == 1]["FirstPokeCorrect"] == 1)
            / len(Ndf[Ndf["SwitchSide"] == 1])
        )
        # percentage of correct trials on stay trials with stimulation
        StayStim = (
            100
            * np.sum(Odf[Odf["SwitchSide"] == 0]["FirstPokeCorrect"] == 1)
            / len(Odf[Odf["SwitchSide"] == 0])
        )
        # percentage of correct trials on switch trials with stimulation
        SwitchStim = (
            100
            * np.sum(Odf[Odf["SwitchSide"] == 1]["FirstPokeCorrect"] == 1)
            / len(Odf[Odf["SwitchSide"] == 1])
        )
        # fill the dataframe
        SessionDF = pd.DataFrame(
            {
                "SessionTime": np.repeat(session, 4),
                "Condition": np.array(
                    [
                        "Normal_Stay",
                        "Normal_Switch",
                        "Opto_Stay",
                        "Opto_Switch",
                    ]
                ),
                "PercCorrect": np.array(
                    [StayNoStim, SwitchNoStim, StayStim, SwitchStim]
                ),
            }
        )
        # append it to list
        sessionsInfo.append(SessionDF)

    # merge into a single df and return
    return pd.concat(sessionsInfo, ignore_index=True)


# Analyze this with the trial side as well
def AnalyzeSwitchTrials_for_sides(df):
    # df is a dataframe containing the following columns:
    # 'SwitchSide'
    # 'FirstPokeCorrect'
    # 'SessionTime'
    # 'TrialSide'
    # it returns a different dataframe with information grouped for a bar plot

    # get info for the sessions
    sessionsID = pd.unique(df["SessionTime"])
    # initialize list to hold dataframes
    sessionsInfo = []

    # fill the new dataframe with info for each session
    for session in sessionsID:
        # get the dataframe for that session
        Sdf = df[df["SessionTime"] == session]
        # split the dataset into opto and normal
        Ndf = Sdf[Sdf["TrialSide"] == 1]
        Odf = Sdf[Sdf["TrialSide"] == 2]
        # percentage of correct trials on stay trials without stimulation
        StayNoStim = (
            100
            * np.sum(Ndf[Ndf["SwitchSide"] == 0]["FirstPokeCorrect"] == 1)
            / len(Ndf[Ndf["SwitchSide"] == 0])
        )
        # percentage of correct trials on switch trials without stimulation
        SwitchNoStim = (
            100
            * np.sum(Ndf[Ndf["SwitchSide"] == 1]["FirstPokeCorrect"] == 1)
            / len(Ndf[Ndf["SwitchSide"] == 1])
        )
        # percentage of correct trials on stay trials with stimulation
        StayStim = (
            100
            * np.sum(Odf[Odf["SwitchSide"] == 0]["FirstPokeCorrect"] == 1)
            / len(Odf[Odf["SwitchSide"] == 0])
        )
        # percentage of correct trials on switch trials with stimulation
        SwitchStim = (
            100
            * np.sum(Odf[Odf["SwitchSide"] == 1]["FirstPokeCorrect"] == 1)
            / len(Odf[Odf["SwitchSide"] == 1])
        )
        # fill the dataframe
        SessionDF = pd.DataFrame(
            {
                "SessionTime": np.repeat(session, 4),
                "TrialSide": np.array(
                    ["Left_Stay", "Left_Switch", "Right_Stay", "Right_Switch"]
                ),
                "PercCorrect": np.array(
                    [StayNoStim, SwitchNoStim, StayStim, SwitchStim]
                ),
            }
        )
        # append it to list
        sessionsInfo.append(SessionDF)

    # merge into a single df and return
    return pd.concat(sessionsInfo, ignore_index=True)


# function to process the data of an experiment
# for psychometric performance plots:
def PP_ProcessExperiment(SessionData, bootstrap=None, error_bars=None):
    # SessionData is a dataframe that needs to have the following column names:
    # 'TrialHighPerc'
    # 'FirstPoke'

    diffs = np.array(SessionData["TrialHighPerc"])
    choices = np.array(SessionData["FirstPoke"])

    # Calculate psychometric performance parameters
    PsyPer = PsychPerformance(trialsDif=diffs, sideSelected=choices)
    # predict data
    predictDif = np.linspace(1, 100, 2000).reshape(-1, 1)
    if PsyPer:
        predictPer = 100 * PsyPer["Logit"].predict_proba(predictDif)[:, 1]

        # Calculate the error bars if asked to
        if error_bars is not None:
            EBdata = SessionData.groupby(by=error_bars).apply(getEBdata)
            # flatten the lists
            EB_diffs_flat = list(chain(*[x["Difficulty"] for x in EBdata]))
            EB_perfs_flat = list(chain(*[x["Performance"] for x in EBdata]))
            # calculate error bars for each difficulty
            Std_list = [
                np.std(list(compress(EB_perfs_flat, EB_diffs_flat == dif)))
                for dif in PsyPer["Difficulty"]
            ]
        else:
            Std_list = np.nan
    else:  # needed for the return
        predictPer = np.nan
        Std_list = np.nan

    # Bootstrap on fake data (generated inside the bootstrap function)
    fakePredictions = np.nan
    if bootstrap is not None:
        np.random.seed(12233)  # fixed random seed for reproducibility
        if PsyPer:
            fakePredictions = BootstrapPerformances(
                trialsDif=diffs,
                sideSelected=choices,
                ntimes=bootstrap,
                prediction_difficulties=predictDif,
            )

    # return what is needed for the plot
    return predictDif, PsyPer, fakePredictions, predictPer, Std_list


def getEBdata(SessionData):
    # SessionData is a dataframe that needs to have the following column names:
    # 'TrialHighPerc'
    # 'FirstPoke'

    diffs = np.array(SessionData["TrialHighPerc"])
    choices = np.array(SessionData["FirstPoke"])

    PsyPer = PsychPerformance(trialsDif=diffs, sideSelected=choices)

    return PsyPer


def timeDifferences(listOfDates):
    """
    Return the absolute time, in days, of elements in a list of dates,
    related to the first
    Assumes data is in order (would return negative values otherwise)
    :param listOfDates: list of size X of dates. Format: YYYYMMDD_HHMMSS
    :return: array of size X of absolute time
    """

    if len(listOfDates) == 0:
        return []

    abstimeList = []
    for date in listOfDates:
        strList = [
            int(date[0:4]),
            int(date[4:6]),
            int(date[6:8]),
            int(date[9:11]),
            int(date[11:13]),
            int(date[13:15]),
        ]
        intList = list(map(int, strList))
        # Calculate absolute time in days

        multipliers = [365, 30, 1, 1 / 24, 1 / (24 * 60), 1 / (24 * 60 * 60)]
        mulList = [a * b for a, b in zip(intList, multipliers)]
        abstime = sum(mulList)
        abstimeList.append(abstime)

    diftime = np.array(abstimeList) - abstimeList[0]

    return diftime


def RBias(FirstPokes, FirstPokesCorrect):
    """
    Returns the bias to the right
    FirstPokes is a vector of 1s and 2s (Left or Right), indicating
    the poked port
    FirstPokesCorrect is a 0 and 1 vector (wrong or correct poke)
    Both could have NaN values

    Returns from -1 to 1. 0 Being not biased, 1 being Right-biased, and
    -1 being left-biased. It is a conservative function. E.g, in a 50-50
    trial chance, and being totally biased to one side, only half of the
    trials would be wrong, so the function would output +/-0.5.

    Correct trials based on proportion of wrong pokes
    Determine the proportion of wrong pokes to the right side
    """
    WrongSides = FirstPokes[FirstPokesCorrect == 0]
    if len(WrongSides) < 1:
        RBias = 0
    else:
        WrongSideProportion = len(WrongSides) / len(FirstPokes)  # from 0 to 1
        WrongRightsProportion = (
            WrongSideProportion * np.nansum(WrongSides == 2) / len(WrongSides)
        )
        WrongLeftsProportion = (
            WrongSideProportion * np.nansum(WrongSides == 1) / len(WrongSides)
        )

        RBias = WrongRightsProportion - WrongLeftsProportion
    return RBias


def CalculateRBiasWindow(FirstPokes, FirstPokesCorrect, Window):
    """Calculates RBias over the lenght of the vectors FirstPokes and
    FirstPokesCorrect using a Window. Returns vector of same lenght"""
    # Create empty vector
    RBiasVector = np.empty(len(FirstPokes))
    RBiasVector[:] = np.nan
    for i in range(Window, len(FirstPokes)):
        win = range((i - Window), i)
        RBiasVector[i] = RBias(FirstPokes[win], FirstPokesCorrect[win])

    return RBiasVector


# calculate the number of times they go to the middle (anxiousness?)
def CalculateMidPokes(df):
    return np.sum(
        df["TrialEvents"]["Port2In"] <= df["TrialStates"]["WaitForResponse"][0]
    )
    # this might fail if WaitForResponse is empty...


# quantify how long they wait in the middle
def MidPortWait(df):
    timeOut = df["TrialStates"]["WaitForResponse"].astype("float32")[0]
    PortIn = df["TrialEvents"]["Port2In"]
    # sometimes this is an integer (rarely)
    if isinstance(PortIn, int):
        PortIn = float(PortIn)
    if not isinstance(PortIn, float):
        PortIn = PortIn.astype("float32")  # does not work for int
        PortInIdx = np.where(PortIn < timeOut)[0][-1]
        PortInTime = PortIn[PortInIdx]
    else:
        PortInTime = PortIn

    PortTime = timeOut - PortInTime
    return PortTime


def CalculateTrialInitiationTime(df):
    # quantify the time they take to initiate a trial
    # (from trialstart to center poke in)
    # the first time they poke
    try:
        return float(df.TrialEvents["Port2In"][0])
    except Exception:
        return float("NaN")


def AnalyzePercentageByDay(rdf):
    # df is a dataframe containing the following columns:
    # 'FirstPokeCorrect'
    # 'TrainingDay'
    # 'AnimalID'
    # 'Protocol'
    # 'ExperimentalGroup'
    # it returns a different dataframe with information grouped for a bar plot
    AnimalIDs = pd.unique(rdf["AnimalID"])
    animalsInfo = []
    for animalid in AnimalIDs:
        df = rdf[rdf["AnimalID"] == animalid]
        # get info for the sessions
        TrainingDays = pd.unique(df["TrainingDay"])
        # initialize value for cumulative trials
        CumTrials = 0
        # fill the new dataframe with info for each session
        for session in TrainingDays:
            # get the dataframe for that session
            Sdf = df[df["TrainingDay"] == session]
            # protocol and ExperimentalGroup
            prot = Sdf.Protocol.iloc[0]
            inj = Sdf.ExperimentalGroup.iloc[0]
            # percentage of correct trials
            PercCorrect = 100 * np.sum(Sdf["FirstPokeCorrect"]) / len(Sdf)
            # number of trials per session
            NumOfTrials = len(Sdf)
            # cumulative trials
            CumTrials = CumTrials + NumOfTrials
            # fill the dataframe
            SessionDF = pd.DataFrame(
                {
                    "AnimalID": animalid,
                    "SessionTime": session,
                    "PercCorrect": np.array([PercCorrect]),
                    "NumberOfTrials": NumOfTrials,
                    "CumulativeTrials": CumTrials,
                    "Protocol": prot,
                    "ExperimentalGroup": inj,
                }
            )
            # append it to list
            animalsInfo.append(SessionDF)
    # merge into a single df and return

    return pd.concat(animalsInfo, ignore_index=True)


def get_new_files(filelist, existing_dates):
    """
    Compares dates in files to a datetime dataset to check for existing data
        :param filelist: list of full paths to bpod files
        :type filelist: list of strings
        :param existing_dates: time objects in datetime format
        :returns: subset of filelist
    """
    filenames = [ntpath.basename(x) for x in filelist]
    dates = BpodDatesToTime(ParseForTimes(filenames))
    dates_formatted = [str(i) for i in dates]
    existing_dates_formatted = [str(i) for i in existing_dates]
    new_dates = list(set(dates_formatted) - set(existing_dates_formatted))
    new_idx = [i for i, n in enumerate(dates_formatted) if n in new_dates]
    new_files = [filelist[i] for i in new_idx]
    return new_files


def split_files_into_old_and_new(filelist, existing_dates):
    """
    Compares dates in files to a datetime dataset to split them
    into new files and old files
        :param filelist: list of full paths to bpod files
        :type filelist: list of strings
        :param existing_dates: time objects in datetime format
        :returns: two subsets of filelist
    """
    # files with a new date
    dif_files = get_new_files(filelist, existing_dates)
    # compare dates and split
    # idx of old_files
    filenames = [ntpath.basename(x) for x in dif_files]
    dates = BpodDatesToTime(ParseForTimes(filenames))
    old_idx = [
        i
        for i, n in enumerate(dates)
        if n < existing_dates.max().to_pydatetime()
    ]
    # split
    old_files = [dif_files[i] for i in old_idx]
    new_files = [
        dif_files[i] for i in list(set(range(len(dif_files))) - set(old_idx))
    ]

    return old_files, new_files


def perf_window_calculator(df, window):
    """
    Calculate the performance of the last X trials

    """
    firstpokecorrect = df["FirstPokeCorrect"]  # 0s and 1s
    # create empty vector of the same size
    perf_window = np.full(len(firstpokecorrect), np.nan)
    for i in range(window - 1, len(perf_window)):
        perf_window[i] = (
            np.nansum(firstpokecorrect[i - window + 1 : i + 1]) / window * 100
        )
    return perf_window


# calculate the trials per minute that animals do by fitting a line
def trials_per_minute(trial_index, trial_start_timestamp):
    """
    function to calculate the speed of the mouse in trials per minute
    param trial_index: pandas.core.series.Series with the trial index
    param trial_start_timestamp: pandas.core.series.Series with the trial
    start time in seconds
    returns a value which is the trials per minute
    """
    lrmodel = LinearRegression().fit(
        trial_index[:, np.newaxis], trial_start_timestamp
    )

    return 60 * 1 / lrmodel.coef_[0]


def speed_window_calculator(df, window):
    """
    Calculate the speed over X trials

    """
    trial_index = df.TrialIndex
    trial_start_timestamp = df.TrialStartTimestamp
    # create empty vector of the same size
    speed_window = np.full(len(trial_index), np.nan)
    for i in range(int(window / 2) - 1, len(speed_window) - int(window / 2)):
        win_idx_low = i - int(window / 2) + 1
        win_idx_high = i + int(window / 2)
        speed_window[i] = trials_per_minute(
            trial_index[win_idx_low:win_idx_high],
            trial_start_timestamp[win_idx_low:win_idx_high],
        )
    return speed_window


def itis_calculator(df):
    # df is a behavioural dataframe

    # find inter-trial-intervals
    itis = np.diff(df.TrialStartTimestamp)
    # append a 0 at the beginning so it matches the trial indexes
    # how long did the mouse take to initiate this trial from the previous?
    itis = np.insert(itis, 0, 0)

    return itis


def find_disengaged_trials(itis):
    # itis is a vector of inter trial intervals
    # this function returns indexes

    disengaged_indexes = np.where(itis > 3 * np.median(itis))

    return disengaged_indexes


def sigmoid_func(x, slope, bias, upper_lapse, lower_lapse):
    return (upper_lapse - lower_lapse) / (
        1 + np.exp(-slope * (x - bias))
    ) + lower_lapse


def linear_func(x, beta, alpha):
    return beta * x + alpha


def fit_custom_sigmoid(difficulty, performance):
    # scale the data
    xdatasc = (difficulty - difficulty.mean()) / difficulty.std()
    ydatasc = performance / 100

    cost_func = lambda x: np.mean(
        np.abs(sigmoid_func(xdatasc, x[0], x[1], x[2], x[3]) - ydatasc)
    )

    res = opt.minimize(cost_func, [-3, 0, 1, 0])

    # rescale
    slope = res.x[0] / difficulty.std()
    bias = res.x[1] * difficulty.std() + difficulty.mean()
    upper_lapse = res.x[2] * 100
    lower_lapse = res.x[3] * 100

    return slope, bias, upper_lapse, lower_lapse


def get_random_optolike_choices(df, n_times=100):
    """
    gets a dataframe that has optostimulated trials,
    and returns, per each difficulty,
    choices sampled randomly from the non-stimulated trials, n_times
    """
    normal_df, opto_df = splitOpto(df)
    fake_opto_side_sel_samples = np.zeros(
        (n_times, len(opto_df["SideSelected"]))
    )

    for i in range(n_times):
        fake_opto_side_sel = np.empty_like(opto_df["SideSelected"])
        for curr_diff in np.unique(opto_df["Difficulty"]):
            diff_opto_mask = opto_df["Difficulty"] == curr_diff
            diff_normal_mask = normal_df["Difficulty"] == curr_diff
            population = normal_df["SideSelected"][diff_normal_mask]
            fake_opto_side_sel[diff_opto_mask] = np.random.choice(
                population, sum(diff_opto_mask)
            )
        fake_opto_side_sel_samples[i] = fake_opto_side_sel

    return fake_opto_side_sel_samples


def get_mean_and_std_of_random_optolike_choices(df, n_times=100):

    # deprecated

    """
    gets a dataframe that has optostimulated trials, and
    outputs, per difficulty, the mean and the std of
    choices sampled randomly from the non-stimulated trials, n_times
    """
    normal_df, opto_df = splitOpto(df)

    available_difficulties = np.unique(opto_df["Difficulty"])
    random_means = np.zeros_like(available_difficulties)
    random_std = np.zeros_like(available_difficulties)

    for k, curr_diff in enumerate(available_difficulties):
        diff_opto_mask = opto_df["Difficulty"] == curr_diff
        diff_normal_mask = normal_df["Difficulty"] == curr_diff
        population = normal_df["SideSelected"][diff_normal_mask]
        if len(population) == 0:
            sys.exit("No normal trials with that difficulty")
        fake_opto_side_sel_list = np.zeros(n_times)
        for i in range(n_times):
            fake_opto_side_sel_list[i] = np.nanmean(
                np.random.choice(population, sum(diff_opto_mask))
            )
        random_means[k] = np.nanmean(fake_opto_side_sel_list)
        random_std[k] = np.nanstd(fake_opto_side_sel_list)

    df_to_return = pd.DataFrame(
        {
            "Difficulty": available_difficulties,
            "Mean_of_choice": 100 * (random_means - 1),
            "Std_of_choice": 100 * random_std,
        }
    )

    return df_to_return


def get_choices(sideSelected, trialsDif):
    """
    returns mean of choices per difficulty
    """
    # Calculate performance
    # Initialize values
    difficulty = np.unique(trialsDif[~np.isnan(trialsDif)])
    choice_mean = np.full(len(difficulty), np.nan)
    for i in range(len(difficulty)):
        if np.nansum(sideSelected[trialsDif == difficulty[i]]) > 0:
            choice_mean[i] = 100 * (
                np.nanmean(sideSelected[trialsDif == difficulty[i]]) - 1
            )

        else:
            choice_mean[i] = np.nan

    return difficulty, choice_mean


def generate_eg(list_size, prob, labs):
    # function to generate a list of experimental groups randomly
    ltr = []
    for i in range(list_size):
        if random.random() < prob:
            ltr.append(labs[0])
        else:
            ltr.append(labs[1])
    return ltr


def get_shuffled_means_difference_df(df_colsel, hue_order, nsh):
    # get the experimental group for each animal
    exp_gr = [
        df_colsel[df_colsel.AnimalID == x].ExperimentalGroup.unique()[0]
        for x in df_colsel.AnimalID.unique()
    ]
    # get the number of mice
    n_an = len(exp_gr)
    # get the probability of a mouse to be a control for this dataset
    cb_prob = sum([x == hue_order[0] for x in exp_gr]) / n_an
    # set random seed
    np.random.seed(124321)
    # calculate the differences of means by resampling
    shuff_res = []

    for i in range(nsh):
        # shuffle the list of groups by assigning a probability for each mouse
        # to be in a group based on the real ratio
        exp_grs = generate_eg(n_an, cb_prob, hue_order)
        # create a diccionary
        egs_dict = dict(zip(df_colsel.AnimalID.unique(), exp_grs))
        # create a new column with the shuffled group
        df_colsel["egs"] = [egs_dict[x] for x in df_colsel.AnimalID]
        # calculate the differences and append
        shuff_res.append(
            df_colsel[df_colsel.egs == hue_order[1]]
            .groupby("TrialIndexBinned")
            .mean()["Performance"]
            - df_colsel[df_colsel.egs == hue_order[0]]
            .groupby("TrialIndexBinned")
            .mean()["Performance"]
        )
        update_progress(i / nsh)

    update_progress(1)

    # return in a data frame format
    return pd.concat(shuff_res)


def get_shuffled_means_difference_global_significance(
    df_colsel, shrdf, quants_to_test, nsh, hue_order
):
    # get the experimental group for each animal
    exp_gr = [
        df_colsel[df_colsel.AnimalID == x].ExperimentalGroup.unique()[0]
        for x in df_colsel.AnimalID.unique()
    ]
    # get the number of mice
    n_an = len(exp_gr)
    # get the probability of a mouse to be a control for this dataset
    cb_prob = sum([x == hue_order[0] for x in exp_gr]) / n_an
    # create an empty array to store results
    global_sig = np.empty((nsh, len(quants_to_test)), dtype=bool)
    # loop over shuffle data
    for i in range(nsh):
        # shuffle the list of groups by assigning a probability for each mouse
        # to be in a group based on the real ratio
        exp_grs = generate_eg(n_an, cb_prob, hue_order)
        # create a diccionary
        egs_dict = dict(zip(df_colsel.AnimalID.unique(), exp_grs))
        # create a new column with the shuffled group
        df_colsel["egs"] = [egs_dict[x] for x in df_colsel.AnimalID]
        # calculate the differences
        sh_dif = (
            df_colsel[df_colsel.egs == hue_order[1]]
            .groupby("TrialIndexBinned")
            .mean()["Performance"]
            - df_colsel[df_colsel.egs == hue_order[0]]
            .groupby("TrialIndexBinned")
            .mean()["Performance"]
        )
        # for each quantile band, what percentages of lines cross at any point
        for k, q in enumerate(quants_to_test):
            global_sig[i, k] = any(
                np.logical_or(
                    sh_dif > shrdf.groupby("TrialIndexBinned").quantile(q),
                    sh_dif < shrdf.groupby("TrialIndexBinned").quantile(1 - q),
                )
            )

        update_progress(i / nsh)
    update_progress(1)

    return global_sig


def get_random_choices_for_optostimulation(df, ntimes):

    data = np.empty([len(pd.unique(df["SessionID"])), 3], dtype=object)

    for i, session in enumerate(pd.unique(df["SessionID"])):
        # generate the random dataset, and save it to a general
        # dataframe for later use
        session_df = df[df["SessionID"] == session]
        roc = get_random_optolike_choices(df=session_df, n_times=ntimes)
        _, odf = splitOpto(session_df)
        roc_ds = np.apply_along_axis(
            get_choices, 1, roc, trialsDif=odf["Difficulty"]
        )
        avail_diffs = roc_ds[0, 0, :]
        rand_choices_ds = roc_ds[:, 1, :]
        # fill
        data[i] = [session, avail_diffs, rand_choices_ds]

        update_progress(
            i / len(pd.unique(df["SessionID"])), head="Generating dataset"
        )

    random_opto_df = pd.DataFrame(
        data, columns=["SessionID", "Difficulties", "Random_choices"]
    )
    update_progress(1)

    return random_opto_df


def calculate_differences_with_random_optostimulation(
    otl_df, random_opto_df, n_times
):
    # Calculate the differences between the random choices and the
    # opto ones to measure the bias

    random_opto_df["bias"] = None
    random_opto_df["bias_mean"] = None
    random_opto_df["bias_std"] = None

    tot_sess = len(pd.unique(random_opto_df["SessionID"]))

    for sc, session in enumerate(pd.unique(random_opto_df["SessionID"])):
        session_idx = random_opto_df.index[
            random_opto_df.SessionID == session
        ].item()
        # get data for opto
        opto_df = otl_df[
            (otl_df["SessionID"] == session) & (otl_df["OptoStim"] == 1)
        ]
        trialsDif = np.array(opto_df["TrialHighPerc"])
        sideSelected = np.array(opto_df["FirstPoke"])
        difficulty, opto_perf = get_choices(sideSelected, trialsDif)

        # get data for the shuffles
        updown_list = np.empty(n_times)
        for i in range(n_times):
            y_vals = random_opto_df.loc[session_idx].Random_choices[i]
            # calculate difference and normalize
            updown_list[i] = np.sum(y_vals - opto_perf) / len(difficulty)

        random_opto_df.at[session_idx, "bias"] = updown_list
        random_opto_df.at[session_idx, "bias_mean"] = np.nanmean(updown_list)
        random_opto_df.at[session_idx, "bias_std"] = np.nanstd(updown_list)

        update_progress((sc + 1) / tot_sess, head="Getting differences")

    return random_opto_df


def add_info_and_contralateral_bias(oal_df, random_opto_df):
    # add the side in which stimulation happened,
    # and translate the bias to contralateral / ipsilateral
    random_opto_df["stimulated_side"] = None
    random_opto_df["contralateral_bias"] = None
    # Get a column with the mouse name
    random_opto_df["AnimalID"] = None
    # And the type of mouse
    random_opto_df["Genotype"] = None

    tot_sess = len(pd.unique(random_opto_df["SessionID"]))

    for sc, session in enumerate(pd.unique(random_opto_df["SessionID"])):
        session_idx = random_opto_df.index[
            random_opto_df.SessionID == session
        ].item()
        # get information
        stim_side = (
            oal_df[oal_df["SessionID"] == session].Stimulation.unique().item()
        )
        # add info
        random_opto_df.at[session_idx, "stimulated_side"] = stim_side
        mouse_name = random_opto_df.loc[session_idx].SessionID.split(" ")[0]
        random_opto_df.at[session_idx, "AnimalID"] = mouse_name
        random_opto_df.at[session_idx, "Genotype"] = mouse_name.split("-")[0]

        # bias is the normal minus the opto.
        # This means that a positive value is a leftwards bias
        # of the opto trials
        # this is good as a contralateral bias for the trials in which
        # optostimulation occurs in the right side
        # flip the sign of the ones where stimulation happens on the left
        if stim_side == "Right":
            random_opto_df.at[
                session_idx, "contralateral_bias"
            ] = random_opto_df.at[session_idx, "bias_mean"]
        elif stim_side == "Left":
            random_opto_df.at[
                session_idx, "contralateral_bias"
            ] = -random_opto_df.at[session_idx, "bias_mean"]
        elif stim_side == "Both":
            random_opto_df.at[session_idx, "contralateral_bias"] = np.nan
        else:
            print("Something wrong")

        update_progress((sc + 1) / tot_sess, head="Adding info")

    return random_opto_df


def get_random_dataframe_for_optostimulation(oal_df, n_times):

    # Generate random optostimulation choices for every session
    random_opto_df = get_random_choices_for_optostimulation(oal_df, n_times)

    # Calculate differences with the stimulated trials
    random_opto_df = calculate_differences_with_random_optostimulation(
        oal_df, random_opto_df, n_times
    )

    # add the side in which stimulation happened, mouse name and genotype,
    # and translate the bias to contralateral / ipsilateral
    random_opto_df = add_info_and_contralateral_bias(oal_df, random_opto_df)

    print("Done computing the random dataframe")

    return random_opto_df


def difficulty_for_bias(mouse_line, stimulated_side):
    # function to determine which difficulty to look at
    # create a logic table in order to find which difficulty
    # to look at when calculating the bias
    # this depends on the contingency, on the mouse line,
    # and on the the fiber placement

    # expected movements given the sound (difficulty)
    set_contingency = {"Right": 2.0, "Left": 98.0}

    # expectation (and observation) of bias
    bias_expectation = {"D1opto": "ipsi", "D2opto": "contra"}

    # logical table for what to expect given ipsi/contra
    # e.g. if you expect an ipsi bias and the fiber is on the right,
    # you wanna look at the left movements
    # -----------------------------
    #           |   Right   Left
    #           |-----------------
    #  ipsi     |   Left    Right
    #  contra   |   Right   Left
    # -----------------------------
    logic_rows = ["ipsi", "contra"]
    logic_cols = ["Right", "Left"]
    logic_table = [["Left", "Right"], ["Right", "Left"]]

    expected_bias = bias_expectation[mouse_line]
    table_row = logic_rows.index(expected_bias)
    table_col = logic_cols.index(stimulated_side)
    affected_side = logic_table[table_row][table_col]

    return set_contingency[affected_side]


def fiber_unique_id(panda_series):
    return "_".join([panda_series.AnimalID, panda_series.stimulated_side])


def significance_calculator(panda_series):
    cbe = panda_series.bias_exp
    # get contra value
    if panda_series.stimulated_side == "Left":
        cbe = [-x for x in cbe]

    if panda_series.Genotype == "D1opto":
        n_sig = np.sum([x > 0 for x in cbe])
    if panda_series.Genotype == "D2opto":
        n_sig = np.sum([x < 0 for x in cbe])

    return n_sig / len(cbe)


def get_simplified_dataframe_for_optostimulation(random_opto_df):

    # columns: animal_id | genotype
    #          | session_performance | contralateral_bias_exp
    animal_id_list = []
    genotype_list = []
    session_performance_list = []
    contra_bias_list = []

    fibers = pd.unique(random_opto_df.fiber_id)
    for animal in fibers:
        animal_rdf = random_opto_df[random_opto_df.fiber_id == animal]
        for session in pd.unique(animal_rdf.SessionID):
            session_idx = animal_rdf.index[
                animal_rdf.SessionID == session
            ].item()
            sp = animal_rdf.loc[session_idx].session_performance
            cb = animal_rdf.loc[session_idx].contralateral_bias_exp
            session_performance_list.append(sp)
            contra_bias_list.append(cb)
            genotype_list.append(animal_rdf.loc[session_idx].Genotype)
            animal_id_list.append(animal)

    simplified_df = pd.DataFrame(
        {
            "animal_id": animal_id_list,
            "genotype": genotype_list,
            "session_performance": session_performance_list,
            "contralateral_bias": contra_bias_list,
        }
    )

    return simplified_df


def get_fit_coefs(df):
    x = df.session_performance
    y = df.contralateral_bias
    return np.polyfit(x, y, 1)


def get_dicctionary_of_regressions_optostimulation(
    simplified_df, shuffle_times=100, xs=range(50, 100)
):
    # calculate slopes and generate shuffles of biases
    # per mouse to get the significance for each individual
    # save all in a diccionary
    reg_dicc = {
        "animals": [],
        "genotypes": [],
        "reg_coefs": [],
        "fits": [],
        "predicted_matrices": [],
        "shuffled_coefficients": [],
    }

    for animal in simplified_df.animal_id.unique():
        reg_df = simplified_df[simplified_df.animal_id == animal].copy()
        slope, intercept = get_fit_coefs(reg_df)

        # get a list of coefficients for suffled dataframes
        shuffled_slopes = np.zeros(shuffle_times)
        shuffled_int = np.zeros(shuffle_times)
        # generate a matrix of predictions
        predicted_matrix = np.zeros([shuffle_times, len(xs)])

        for i in range(shuffle_times):
            # shuffle dataframe
            shuffled_df = reg_df.copy()
            np.random.shuffle(shuffled_df.contralateral_bias.values)
            # get coefficients
            shuffled_slopes[i], shuffled_int[i] = get_fit_coefs(shuffled_df)
            # fill matrix
            predicted_matrix[i, :] = shuffled_int[i] + shuffled_slopes[i] * xs

        # fill diccionary
        reg_dicc["animals"].append(animal)
        reg_dicc["genotypes"].append(
            simplified_df[simplified_df.animal_id == animal].genotype.unique()[
                0
            ]
        )
        reg_dicc["reg_coefs"].append([slope, intercept])
        reg_dicc["fits"].append(intercept + slope * xs)
        reg_dicc["predicted_matrices"].append(predicted_matrix)
        reg_dicc["shuffled_coefficients"].append(
            [shuffled_slopes, shuffled_int]
        )

    return reg_dicc


def get_binned_dataframe_for_optostimulation(
    random_opto_df, significance=0.05
):

    # save all in a diccionary
    binned_dict = {
        "genotype": [],
        "performance_window": [],
        "contra_biases": [],
        "contra_bias_mean": [],
        "contra_bias_std": [],
        "contra_bias_upper_percentile": [],
        "contra_bias_lower_percentile": [],
        "n_sessions": [],
        "n_animals": [],
        "significance_expected_bias": [],
        "number_of_significant_sessions": [],
    }

    # binned_window = 10
    performance_windows = [
        [0, 60],
        [60, 65],
        [65, 70],
        [70, 75],
        [75, 80],
        [80, 85],
        [85, 90],
        [90, 100],
    ]

    for genot in ["D1opto", "D2opto"]:
        # subselect the dataframe
        genotdf = random_opto_df[random_opto_df.Genotype == genot].copy()

        for bracket in performance_windows:
            # find sessions belonging to that bracket
            bracket_mask = np.logical_and(
                bracket[0] <= genotdf.session_performance,
                genotdf.session_performance < bracket[1],
            )
            subdf = genotdf[bracket_mask].copy()
            # extract the contralateral biases of random choices
            contralateral_bias_exp_merge = []
            n_sess = subdf.shape[0]
            n_ans = len(np.unique(subdf.AnimalID))
            for i in range(n_sess):
                cbe = subdf.iloc[i].bias_exp
                # get contra value
                if subdf.iloc[i].stimulated_side == "Left":
                    cbe = [-x for x in cbe]
                contralateral_bias_exp_merge.append(cbe)
            # flatten
            contralateral_bias_exp_merge = [
                item
                for sublist in contralateral_bias_exp_merge
                for item in sublist
            ]

            # append to dicc
            binned_dict["genotype"].append(genot)
            binned_dict["performance_window"].append(bracket[1] - 2.5)
            binned_dict["contra_biases"].append(contralateral_bias_exp_merge)
            binned_dict["contra_bias_mean"].append(
                np.mean(contralateral_bias_exp_merge)
            )
            binned_dict["contra_bias_std"].append(
                np.std(contralateral_bias_exp_merge)
            )
            if len(contralateral_bias_exp_merge) > 1:
                binned_dict["contra_bias_upper_percentile"].append(
                    np.percentile(contralateral_bias_exp_merge, 97.5)
                )
                binned_dict["contra_bias_lower_percentile"].append(
                    np.percentile(contralateral_bias_exp_merge, 2.5)
                )
            else:
                binned_dict["contra_bias_upper_percentile"].append(np.nan)
                binned_dict["contra_bias_lower_percentile"].append(np.nan)
            binned_dict["n_sessions"].append(n_sess)
            binned_dict["n_animals"].append(n_ans)
            if genot == "D1opto":
                n_sig = np.sum([x > 0 for x in contralateral_bias_exp_merge])
            if genot == "D2opto":
                n_sig = np.sum([x < 0 for x in contralateral_bias_exp_merge])
            sig = n_sig / len(contralateral_bias_exp_merge)
            binned_dict["significance_expected_bias"].append(sig)

            # calculate the number of sessions that are significant!
            significant_sessions_list = []
            for i in range(n_sess):
                cbe = subdf.iloc[i].bias_exp
                # get contra value
                if subdf.iloc[i].stimulated_side == "Left":
                    cbe = [-x for x in cbe]

                if genot == "D1opto":
                    n_sig = np.sum([x > 0 for x in cbe])
                if genot == "D2opto":
                    n_sig = np.sum([x < 0 for x in cbe])
                sig = n_sig / len(cbe)

                if sig < significance:
                    significant_sessions_list.append(True)
                else:
                    significant_sessions_list.append(False)

            sig = np.sum(significant_sessions_list)
            binned_dict["number_of_significant_sessions"].append(sig)

    # create df
    binned_df = pd.DataFrame(binned_dict)

    # add lower and upper std
    binned_df["lower_std"] = (
        binned_df.contra_bias_mean - binned_df.contra_bias_std
    )
    binned_df["upper_std"] = (
        binned_df.contra_bias_mean + binned_df.contra_bias_std
    )

    return binned_df


def get_general_right_bias(df_one, df_two):
    "returns the general bias to the right, between df_one and df_two"
    # mean choices for each data frame for each difficulty
    tdone = np.array(df_one["TrialHighPerc"])
    ssone = np.array(df_one["FirstPoke"])
    _, perf_one = get_choices(ssone, tdone)
    tdtwo = np.array(df_two["TrialHighPerc"])
    sstwo = np.array(df_two["FirstPoke"])
    _, perf_two = get_choices(sstwo, tdtwo)

    return np.mean(perf_one) - np.mean(perf_two)


def get_random_biases(df, n_times, it, aot):
    # create array
    rblist = np.zeros(n_times)
    for i in range(n_times):
        # shuffle TrialIndexes
        df.TrialIndex = df.TrialIndex.sample(frac=1).values
        # calculate bias
        rblist[i] = get_general_right_bias(
            df[df.TrialIndex < it], df[df.TrialIndex > aot]
        )
    return rblist


def get_dopamine_optostimulation_differences_dataframe(
    dao_df, ini_trials, ao_trials, n_times
):
    # Generate another dataset for every session containing information
    # about the difference between
    # the optostimulated trials and the normal ones,
    # as well as random differences, calculated
    # shuffling the trial indexes
    BRS = ["tStr", "NAc"]
    PS = ["Left", "Right"]
    PI = ["Center", "Side"]
    CondList = [
        (dao_df["TrialIndex"] < ini_trials),
        (dao_df["TrialIndex"] > ao_trials),
    ]

    cols = [
        "AnimalID",
        "SessionID",
        "Ntrials",
        "Protocol",
        "Stim",
        "FiberSide",
        "FiberArea",
        "StimSide",
        "StimPort",
        "Contralateral",
        "InitialBias",
        "Bias",
        "BiasToStimPort",
        "RandomBiases",
        "RandomBiasMean",
        "RandomBiasStd",
    ]
    data = np.empty(
        [len(pd.unique(dao_df["SessionID"])), len(cols)], dtype=object
    )

    for i, sessionid in enumerate(pd.unique(dao_df["SessionID"])):
        # get dataframe of the session
        session_df = dao_df[dao_df["SessionID"] == sessionid].copy()
        # get animal name
        animalid = session_df.AnimalID.unique()[0]
        # get number of trials
        ntrials = session_df.shape[0]
        # protocol
        protocol = session_df.Protocol.unique()[0]
        # is it a stimulated session?
        stim = session_df.Stimulation.unique()[0] != "NoStimulation"
        # which fiber was plugged in
        fiberside = session_df.Stimulation.unique()[0]
        # which brain area is this fiber over
        fiberarea = BRS[int(session_df.iloc[0].FullGUI["FiberLocation"]) - 1]
        # which one of the side ports, or trial type, was stimulated
        stimside = PS[int(session_df.iloc[0].FullGUI["JOPSide"]) - 1]
        # in which one of the ports did stimulation occurred
        stimport = PI[int(session_df.iloc[0].FullGUI["OptoState"]) - 1]
        # is the fiber contralateral to the port
        contralateral = True
        if (fiberside == stimside) or fiberside == "Both":
            contralateral = False
        # what is the initial bias of the mouse in trials before stimulation
        ini_sess = session_df[session_df.TrialIndex < ini_trials].copy()
        initialbias = np.mean(
            get_choices(ini_sess["FirstPoke"], ini_sess["TrialHighPerc"])[1]
        )
        # what is the total bias of that session after opto
        bias = get_general_right_bias(
            session_df[CondList[1]], session_df[CondList[0]]
        )
        # is this bias positive towards the stimulated port?
        if stimside == "Right":
            biastostimport = bias
        if stimside == "Left":
            biastostimport = -bias
        # calculate random biases
        randombiases = get_random_biases(
            session_df, n_times, ini_trials, ao_trials
        )
        # random mean
        randombiasmean = np.mean(randombiases)
        # random std
        randombiasstd = np.std(randombiases)

        # fill
        data[i] = [
            animalid,
            sessionid,
            ntrials,
            protocol,
            stim,
            fiberside,
            fiberarea,
            stimside,
            stimport,
            contralateral,
            initialbias,
            bias,
            biastostimport,
            randombiases,
            randombiasmean,
            randombiasstd,
        ]

        update_progress(i / len(pd.unique(dao_df["SessionID"])))

    # create dataframe
    opto_df = pd.DataFrame(data, columns=cols)
    update_progress(1)

    return opto_df


def find_indexes_of_repeated_cases(opto_df_sel, same_columns):
    # Find indexes of repeated cases
    equal_indexes = []

    for index in opto_df_sel.index:
        data = opto_df_sel.loc[index][same_columns].values
        i_list = []
        for i in opto_df_sel.index:
            if np.array_equal(data, opto_df_sel.loc[i][same_columns].values):
                i_list.append(i)
        if len(i_list) > 1:
            if i_list not in equal_indexes:
                equal_indexes.append(i_list)

    return equal_indexes


def merge_repeated_cases_for_dopamine_optostimulation(opto_df_sel):

    # Find indexes of repeated cases
    same_columns = [
        "AnimalID",
        "FiberSide",
        "FiberArea",
        "StimSide",
        "StimPort",
    ]
    equal_indexes = find_indexes_of_repeated_cases(opto_df_sel, same_columns)

    # Combine those cases
    for case in equal_indexes:
        sub_df = opto_df_sel.loc[case].copy()
        # create new instance to add to the dataframe,
        # initiating it in the first index of the set
        new_element = sub_df.iloc[0].copy()
        # change relevant values
        new_element.SessionID = "merge"
        new_element.Ntrials = np.mean(sub_df.Ntrials.values)
        new_element.Protocol = "merge"
        new_element.InitialBias = np.nan
        new_element.Bias = np.nan
        new_element.BiasToStimPort = np.mean(sub_df.BiasToStimPort.values)
        new_element.RandomBiases = np.concatenate(sub_df.RandomBiases.values)
        new_element.RandomBiasMean = np.mean(new_element.RandomBiases)
        new_element.RandomBiasStd = np.std(new_element.RandomBiases)
        # remove old indexes
        opto_df_sel.drop(case, inplace=True)
        # add new row
        opto_df_sel = opto_df_sel.append(new_element)
    opto_df_sel.sort_index(inplace=True)

    return opto_df_sel
