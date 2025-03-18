import React from 'react';
import TennisPredictor from './components/TennisPredictor';
import './styles/main.css';

function App() {
  return (
    <div className="App">
      <header className="App-header">
        <h1>🎾 Predictor de Partidos de Tenis</h1>
      </header>
      <main>
        <TennisPredictor />
      </main>
      <footer>
        <p>© 2023 Tennis Predictor - Proyecto de Machine Learning</p>
      </footer>
    </div>
  );
}

export default App;