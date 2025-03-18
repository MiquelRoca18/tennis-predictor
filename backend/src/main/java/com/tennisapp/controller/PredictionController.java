package com.tennisapp.controller;

import com.tennisapp.model.MatchData;
import com.tennisapp.model.PredictionResponse;
import com.tennisapp.service.PredictionService;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.MediaType;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

@RestController
@RequestMapping("/api/predictions")
@CrossOrigin(origins = "*")
public class PredictionController {

    @Autowired
    private PredictionService predictionService;

    @PostMapping(
        consumes = MediaType.APPLICATION_JSON_VALUE,
        produces = MediaType.APPLICATION_JSON_VALUE
    )
    public ResponseEntity<PredictionResponse> predict(@RequestBody MatchData matchData) {
        PredictionResponse response = predictionService.predictWinner(matchData);
        return ResponseEntity.ok(response);
    }
}