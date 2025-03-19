package com.tennisapp.model;

import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;

import javax.persistence.Entity;
import javax.persistence.GeneratedValue;
import javax.persistence.GenerationType;
import javax.persistence.Id;

@Data
@NoArgsConstructor
@AllArgsConstructor
@Entity
public class MatchData {
    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;
    
    private String player1;
    private String player2;
    private int ranking1;
    private int ranking2;
    private double winrate1;
    private double winrate2;
    private String surface;
    
    // Si no estás usando Lombok o no está funcionando correctamente,
    // agrega los getters y setters manualmente:
    
    public String getPlayer1() {
        return player1;
    }
    
    public void setPlayer1(String player1) {
        this.player1 = player1;
    }
    
    public String getPlayer2() {
        return player2;
    }
    
    public void setPlayer2(String player2) {
        this.player2 = player2;
    }
    
    public int getRanking1() {
        return ranking1;
    }
    
    public void setRanking1(int ranking1) {
        this.ranking1 = ranking1;
    }
    
    public int getRanking2() {
        return ranking2;
    }
    
    public void setRanking2(int ranking2) {
        this.ranking2 = ranking2;
    }
    
    public double getWinrate1() {
        return winrate1;
    }
    
    public void setWinrate1(double winrate1) {
        this.winrate1 = winrate1;
    }
    
    public double getWinrate2() {
        return winrate2;
    }
    
    public void setWinrate2(double winrate2) {
        this.winrate2 = winrate2;
    }
    
    public String getSurface() {
        return surface;
    }
    
    public void setSurface(String surface) {
        this.surface = surface;
    }
}