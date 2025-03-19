package com.tennisapp.service;

import com.tennisapp.model.MatchData;
import com.tennisapp.model.PredictionResponse;
import com.tennisapp.repository.PredictionRepository;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.http.HttpEntity;
import org.springframework.http.HttpHeaders;
import org.springframework.http.MediaType;
import org.springframework.stereotype.Service;
import org.springframework.web.client.RestTemplate;

import java.util.HashMap;
import java.util.Map;

@Service
public class PredictionServiceImpl implements PredictionService {

    @Autowired
    private RestTemplate restTemplate;

    @Autowired
    private PredictionRepository predictionRepository;

    @Value("${ml.service.url}")
    private String mlServiceUrl;

    @Override
    public PredictionResponse predictWinner(MatchData matchData) {
        // Guardar los datos del partido
        saveMatchData(matchData);
        
        // Preparar los datos para enviar al servicio ML
        Map<String, Object> requestMap = new HashMap<>();
        requestMap.put("player_1", matchData.getPlayer1());
        requestMap.put("player_2", matchData.getPlayer2());
        requestMap.put("ranking_1", matchData.getRanking1());
        requestMap.put("ranking_2", matchData.getRanking2());
        requestMap.put("winrate_1", matchData.getWinrate1());
        requestMap.put("winrate_2", matchData.getWinrate2());
        requestMap.put("surface", matchData.getSurface());

        // Configurar headers
        HttpHeaders headers = new HttpHeaders();
        headers.setContentType(MediaType.APPLICATION_JSON);
        HttpEntity<Map<String, Object>> entity = new HttpEntity<>(requestMap, headers);

        // Enviar petición al servicio ML
        return restTemplate.postForObject(mlServiceUrl + "/predict", entity, PredictionResponse.class);
    }

    @Override
    public MatchData saveMatchData(MatchData matchData) {
        return predictionRepository.save(matchData);
    }
}