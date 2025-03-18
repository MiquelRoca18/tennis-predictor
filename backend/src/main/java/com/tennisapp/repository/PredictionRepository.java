package com.tennisapp.repository;

import com.tennisapp.model.MatchData;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.stereotype.Repository;

@Repository
public interface PredictionRepository extends JpaRepository<MatchData, Long> {
}