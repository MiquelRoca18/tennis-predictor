package com.tennisapp.model;

public class PredictionResponse {
    private String player_1;
    private String player_2;
    private String predicted_winner;
    private double probability;
    
    // Getters y setters con los nombres correctos
    public String getPlayer_1() {
        return player_1;
    }
    
    public void setPlayer_1(String player_1) {
        this.player_1 = player_1;
    }
    
    public String getPlayer_2() {
        return player_2;
    }
    
    public void setPlayer_2(String player_2) {
        this.player_2 = player_2;
    }
    
    public String getPredicted_winner() {
        return predicted_winner;
    }
    
    public void setPredicted_winner(String predicted_winner) {
        this.predicted_winner = predicted_winner;
    }
    
    public double getProbability() {
        return probability;
    }
    
    public void setProbability(double probability) {
        this.probability = probability;
    }
}