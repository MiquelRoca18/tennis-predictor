import React, { useState } from 'react';

const surfaceOptions = [
  { value: 'hard', label: 'Dura' },
  { value: 'clay', label: 'Tierra Batida' },
  { value: 'grass', label: 'Césped' }
];

const TennisForm = ({ onSubmit, isLoading }) => {
  const [formData, setFormData] = useState({
    player1: '',
    player2: '',
    ranking1: '',
    ranking2: '',
    winrate1: '',
    winrate2: '',
    surface: 'hard'
  });

  const handleChange = (e) => {
    const { name, value } = e.target;
    setFormData({
      ...formData,
      [name]: value
    });
  };

  const handleSubmit = (e) => {
    e.preventDefault();
    // Convertir strings a números donde sea necesario y asegurar que los nombres de campos coincidan con el backend
    const data = {
      player1: formData.player1,
      player2: formData.player2,
      ranking1: parseInt(formData.ranking1, 10),
      ranking2: parseInt(formData.ranking2, 10),
      winrate1: parseFloat(formData.winrate1),
      winrate2: parseFloat(formData.winrate2),
      surface: formData.surface
    };
    onSubmit(data);
  };

  return (
    <div className="tennis-form-container">
      <h2>Predicción de Partido de Tenis</h2>
      <form onSubmit={handleSubmit} className="tennis-form">
        <div className="form-group">
          <h3>Jugador 1</h3>
          <div className="input-group">
            <label>Nombre:</label>
            <input
              type="text"
              name="player1"
              value={formData.player1}
              onChange={handleChange}
              required
              placeholder="Ej. Rafael Nadal"
            />
          </div>
          <div className="input-group">
            <label>Ranking ATP:</label>
            <input
              type="number"
              name="ranking1"
              value={formData.ranking1}
              onChange={handleChange}
              required
              min="1"
              placeholder="Ej. 1"
            />
          </div>
          <div className="input-group">
            <label>% Victorias:</label>
            <input
              type="number"
              name="winrate1"
              value={formData.winrate1}
              onChange={handleChange}
              required
              step="0.01"
              min="0"
              max="100"
              placeholder="Ej. 75.5"
            />
          </div>
        </div>

        <div className="form-group">
          <h3>Jugador 2</h3>
          <div className="input-group">
            <label>Nombre:</label>
            <input
              type="text"
              name="player2"
              value={formData.player2}
              onChange={handleChange}
              required
              placeholder="Ej. Novak Djokovic"
            />
          </div>
          <div className="input-group">
            <label>Ranking ATP:</label>
            <input
              type="number"
              name="ranking2"
              value={formData.ranking2}
              onChange={handleChange}
              required
              min="1"
              placeholder="Ej. 2"
            />
          </div>
          <div className="input-group">
            <label>% Victorias:</label>
            <input
              type="number"
              name="winrate2"
              value={formData.winrate2}
              onChange={handleChange}
              required
              step="0.01"
              min="0"
              max="100"
              placeholder="Ej. 80.3"
            />
          </div>
        </div>

        <div className="form-group">
          <div className="input-group">
            <label>Superficie:</label>
            <select 
              name="surface"
              value={formData.surface}
              onChange={handleChange}
              required
            >
              {surfaceOptions.map(option => (
                <option key={option.value} value={option.value}>
                  {option.label}
                </option>
              ))}
            </select>
          </div>
        </div>

        <button 
          type="submit" 
          className="submit-button" 
          disabled={isLoading}
        >
          {isLoading ? 'Prediciendo...' : 'Predecir Ganador'}
        </button>
      </form>
    </div>
  );
};

export default TennisForm;