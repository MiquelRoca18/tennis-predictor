import axios from 'axios';

const API_URL = 'http://localhost:8080/api';

export const predictWinner = async (matchData) => {
  try {
    const response = await axios.post(`${API_URL}/predictions`, matchData, {
      headers: {
        'Content-Type': 'application/json',
        'Accept': 'application/json'
      }
    });
    return response.data;
  } catch (error) {
    console.error('Error predicting winner:', error);
    throw error;
  }
};