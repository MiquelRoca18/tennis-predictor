import React, { useState } from 'react';
import TennisForm from './TennisForm';
import PredictionResult from './PredictionResult';
import { predictWinner } from '../services/api';

const TennisPredictor = () => {
  const [prediction, setPrediction] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);

  const handleSubmit = async (formData) => {
    setIsLoading(true);
    setError(null);
    
    try {
      const result = await predictWinner(formData);
      setPrediction(result);
    } catch (err) {
      setError('Error al obtener la predicción. Por favor, inténtelo de nuevo.');
      console.error(err);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="tennis-predictor-container">
      <TennisForm onSubmit={handleSubmit} isLoading={isLoading} />
      
      {error && <div className="error-message">{error}</div>}
      
      {prediction && <PredictionResult prediction={prediction} />}
    </div>
  );
};

export default TennisPredictor;