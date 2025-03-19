import React from 'react';
import { PieChart, Pie, Cell, ResponsiveContainer, Legend, Tooltip } from 'recharts';

const PredictionResult = ({ prediction }) => {
  if (!prediction) return null;

  const { player_1, player_2, predicted_winner, probability } = prediction;
  const oppositeProb = 1 - probability;
  
  const data = [
    { name: player_1, value: predicted_winner === player_1 ? probability : oppositeProb },
    { name: player_2, value: predicted_winner === player_2 ? probability : oppositeProb }
  ];
  
  const COLORS = ['#0088FE', '#FF8042'];

  return (
    <div className="prediction-result">
      <h2>Resultado de la Predicción</h2>
      <div className="prediction-content">
        <div className="prediction-text">
          <h3>Ganador predicho: <span className="winner">{predicted_winner}</span></h3>
          <p>Probabilidad de victoria: <strong>{(probability * 100).toFixed(2)}%</strong></p>
        </div>
        <div className="prediction-chart">
          <ResponsiveContainer width="100%" height={300}>
            <PieChart>
              <Pie
                data={data}
                cx="50%"
                cy="50%"
                labelLine={false}
                outerRadius={80}
                fill="#8884d8"
                dataKey="value"
                label={({ name, percent }) => `${name} ${(percent * 100).toFixed(0)}%`}
              >
                {data.map((entry, index) => (
                  <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                ))}
              </Pie>
              <Tooltip />
              <Legend />
            </PieChart>
          </ResponsiveContainer>
        </div>
      </div>
    </div>
  );
};

export default PredictionResult;