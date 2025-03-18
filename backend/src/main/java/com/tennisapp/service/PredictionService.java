package com.tennisapp.service;

import com.tennisapp.model.MatchData;
import com.tennisapp.model.PredictionResponse;

public interface PredictionService {
    PredictionResponse predictWinner(MatchData matchData);
    MatchData saveMatchData(MatchData matchData);
}