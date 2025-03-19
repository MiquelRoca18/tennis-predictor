import os

# Leer el contenido actual
with open('utils.py', 'r') as file:
    content = file.read()

# Añadir métodos y atributos faltantes
class_init = """    def __init__(self, data_path=None):
        \"\"\"Inicializa la clase con configuraciones por defecto.\"\"\"
        self.scaler = StandardScaler()
        self.feature_weights = {
            'ranking': 0.3,
            'h2h': 0.2,
            'recent_form': 0.2,
            'surface': 0.15,
            'fatigue': 0.15
        }
        self.data_path = data_path
        self.players_stats = {}
        self.head_to_head_stats = {}
"""

# Reemplazar la inicialización actual
content = content.replace('    def __init__(self):\n        """Inicializa la clase con configuraciones por defecto."""\n        self.scaler = StandardScaler()\n        self.feature_weights = {\n            \'ranking\': 0.3,\n            \'h2h\': 0.2,\n            \'recent_form\': 0.2,\n            \'surface\': 0.15,\n            \'fatigue\': 0.15\n        }', class_init)

# Añadir métodos faltantes
methods_to_add = """
    def build_player_statistics(self):
        \"\"\"Construye estadísticas de jugadores desde los datos.\"\"\"
        if not self.data_path:
            logging.warning("No hay ruta de datos definida para construir estadísticas")
            return
        
        try:
            data = pd.read_csv(self.data_path)
            players = set(data['player_1'].tolist() + data['player_2'].tolist())
            
            for player in players:
                # Partidos como jugador 1
                p1_matches = data[data['player_1'] == player]
                p1_wins = len(p1_matches[p1_matches['winner'] == 0])
                
                # Partidos como jugador 2
                p2_matches = data[data['player_2'] == player]
                p2_wins = len(p2_matches[p2_matches['winner'] == 1])
                
                # Total partidos y victorias
                total_matches = len(p1_matches) + len(p2_matches)
                total_wins = p1_wins + p2_wins
                
                # Estadísticas por superficie
                surface_stats = {}
                for surface in data['surface'].unique():
                    # Superficie como jugador 1
                    p1_surface = p1_matches[p1_matches['surface'] == surface]
                    p1_surface_wins = len(p1_surface[p1_surface['winner'] == 0])
                    
                    # Superficie como jugador 2
                    p2_surface = p2_matches[p2_matches['surface'] == surface]
                    p2_surface_wins = len(p2_surface[p2_surface['winner'] == 1])
                    
                    # Total partidos y victorias en superficie
                    surface_matches = len(p1_surface) + len(p2_surface)
                    surface_wins = p1_surface_wins + p2_surface_wins
                    
                    if surface_matches > 0:
                        surface_win_rate = (surface_wins / surface_matches) * 100
                    else:
                        surface_win_rate = 0
                    
                    surface_stats[surface] = {
                        'matches': surface_matches,
                        'wins': surface_wins,
                        'win_rate': surface_win_rate
                    }
                
                # Obtener ranking promedio
                if 'ranking_1' in data.columns:
                    p1_rankings = p1_matches['ranking_1'].tolist()
                    p2_rankings = p2_matches['ranking_2'].tolist()
                    avg_ranking = np.mean(p1_rankings + p2_rankings) if (p1_rankings + p2_rankings) else 100
                else:
                    avg_ranking = 100
                
                # Calcular tasa de victoria
                win_rate = (total_wins / total_matches) * 100 if total_matches > 0 else 50
                
                # Guardar estadísticas del jugador
                self.players_stats[player] = {
                    'total_matches': total_matches,
                    'total_wins': total_wins,
                    'win_rate': win_rate,
                    'avg_ranking': avg_ranking,
                    'surface_stats': surface_stats
                }
            
            logging.info(f"Estadísticas calculadas para {len(self.players_stats)} jugadores")
            
        except Exception as e:
            logging.error(f"Error construyendo estadísticas de jugadores: {e}")
    
    def build_head_to_head_statistics(self):
        \"\"\"Construye estadísticas head-to-head entre jugadores.\"\"\"
        if not self.data_path:
            logging.warning("No hay ruta de datos definida para construir estadísticas")
            return
        
        try:
            data = pd.read_csv(self.data_path)
            
            for _, row in data.iterrows():
                player1 = row['player_1']
                player2 = row['player_2']
                winner = row['winner']
                
                # Usar tupla ordenada como clave para asegurar consistencia
                player_pair = tuple(sorted([player1, player2]))
                
                # Inicializar si no existe
                if player_pair not in self.head_to_head_stats:
                    self.head_to_head_stats[player_pair] = {
                        'total_matches': 0,
                        'player1_wins': 0,
                        'player2_wins': 0,
                        'player1_win_pct': 0,
                        'player2_win_pct': 0
                    }
                
                # Actualizar estadísticas
                self.head_to_head_stats[player_pair]['total_matches'] += 1
                
                if winner == 0:  # Jugador 1 ganó
                    if player1 == player_pair[0]:
                        self.head_to_head_stats[player_pair]['player1_wins'] += 1
                    else:
                        self.head_to_head_stats[player_pair]['player2_wins'] += 1
                else:  # Jugador 2 ganó
                    if player2 == player_pair[0]:
                        self.head_to_head_stats[player_pair]['player1_wins'] += 1
                    else:
                        self.head_to_head_stats[player_pair]['player2_wins'] += 1
                
                # Calcular porcentajes
                total = self.head_to_head_stats[player_pair]['total_matches']
                p1_wins = self.head_to_head_stats[player_pair]['player1_wins']
                p2_wins = self.head_to_head_stats[player_pair]['player2_wins']
                
                self.head_to_head_stats[player_pair]['player1_win_pct'] = (p1_wins / total) * 100 if total > 0 else 50
                self.head_to_head_stats[player_pair]['player2_win_pct'] = (p2_wins / total) * 100 if total > 0 else 50
            
            logging.info(f"Estadísticas H2H calculadas para {len(self.head_to_head_stats)} pares de jugadores")
            
        except Exception as e:
            logging.error(f"Error construyendo estadísticas head-to-head: {e}")
    
    def transform_match_data(self, match_data):
        \"\"\"
        Transforma los datos de un partido para predecir el resultado.
        
        Args:
            match_data: Diccionario con datos del partido
            
        Returns:
            DataFrame con características para el modelo
        \"\"\"
        try:
            # Extraer datos
            player1 = match_data.get('player_1', match_data.get('player1', ''))
            player2 = match_data.get('player_2', match_data.get('player2', ''))
            surface = match_data.get('surface', 'hard').lower()
            
            # Preparar características
            features = {}
            
            # Ranking
            ranking1 = match_data.get('ranking_1')
            if ranking1 is None and player1 in self.players_stats:
                ranking1 = self.players_stats[player1].get('avg_ranking', 100)
            elif ranking1 is None:
                ranking1 = 100
            
            ranking2 = match_data.get('ranking_2')
            if ranking2 is None and player2 in self.players_stats:
                ranking2 = self.players_stats[player2].get('avg_ranking', 100)
            elif ranking2 is None:
                ranking2 = 100
            
            features['ranking_1'] = ranking1
            features['ranking_2'] = ranking2
            features['ranking_diff'] = ranking1 - ranking2
            
            # Tasas de victoria
            winrate1 = match_data.get('winrate_1')
            if winrate1 is None and player1 in self.players_stats:
                winrate1 = self.players_stats[player1].get('win_rate', 50)
            elif winrate1 is None:
                winrate1 = 50
            
            winrate2 = match_data.get('winrate_2')
            if winrate2 is None and player2 in self.players_stats:
                winrate2 = self.players_stats[player2].get('win_rate', 50)
            elif winrate2 is None:
                winrate2 = 50
            
            features['winrate_1'] = winrate1
            features['winrate_2'] = winrate2
            features['winrate_diff'] = winrate1 - winrate2
            
            # Superficie
            surfaces = {'hard': 0, 'clay': 1, 'grass': 2, 'carpet': 3}
            features['surface_code'] = surfaces.get(surface, 0)
            
            # Estadísticas por superficie
            if player1 in self.players_stats and player2 in self.players_stats:
                p1_surface_stats = self.players_stats[player1].get('surface_stats', {}).get(surface, {})
                p2_surface_stats = self.players_stats[player2].get('surface_stats', {}).get(surface, {})
                
                features['p1_surface_winrate'] = p1_surface_stats.get('win_rate', 50)
                features['p2_surface_winrate'] = p2_surface_stats.get('win_rate', 50)
                features['surface_winrate_diff'] = features['p1_surface_winrate'] - features['p2_surface_winrate']
            else:
                features['p1_surface_winrate'] = 50
                features['p2_surface_winrate'] = 50
                features['surface_winrate_diff'] = 0
            
            # Estadísticas head-to-head
            player_pair = tuple(sorted([player1, player2]))
            if player_pair in self.head_to_head_stats:
                h2h = self.head_to_head_stats[player_pair]
                
                # Determinar quién es quién en las estadísticas H2H
                if player1 == player_pair[0]:
                    features['p1_h2h_wins'] = h2h['player1_wins']
                    features['p2_h2h_wins'] = h2h['player2_wins']
                    features['p1_h2h_winrate'] = h2h['player1_win_pct']
                    features['p2_h2h_winrate'] = h2h['player2_win_pct']
                else:
                    features['p1_h2h_wins'] = h2h['player2_wins']
                    features['p2_h2h_wins'] = h2h['player1_wins']
                    features['p1_h2h_winrate'] = h2h['player2_win_pct']
                    features['p2_h2h_winrate'] = h2h['player1_win_pct']
                
                features['h2h_diff'] = features['p1_h2h_winrate'] - features['p2_h2h_winrate']
            else:
                features['p1_h2h_wins'] = 0
                features['p2_h2h_wins'] = 0
                features['p1_h2h_winrate'] = 50
                features['p2_h2h_winrate'] = 50
                features['h2h_diff'] = 0
            
            # Convertir a DataFrame
            return pd.DataFrame([features])
            
        except Exception as e:
            logging.error(f"Error transformando datos del partido: {e}")
            # Datos mínimos si hay error
            return pd.DataFrame([{
                'ranking_1': match_data.get('ranking_1', 100),
                'ranking_2': match_data.get('ranking_2', 100),
                'winrate_1': match_data.get('winrate_1', 50),
                'winrate_2': match_data.get('winrate_2', 50),
                'surface_code': {'hard': 0, 'clay': 1, 'grass': 2}.get(match_data.get('surface', 'hard').lower(), 0)
            }])
    
    def _get_external_ranking(self, player):
        \"\"\"Método para obtener ranking desde fuentes externas.\"\"\"
        # Esto sería para implementar integración con APIs externas
        # Por ahora devolvemos None y usamos valores predeterminados
        return None
"""

# Añadir los métodos faltantes al final de la clase
class_end_index = content.find("# Funciones mejoradas para compatibilidad con el código antiguo")
content = content[:class_end_index] + methods_to_add + content[class_end_index:]

# Guardar los cambios
with open('utils.py', 'w') as file:
    file.write(content)

print("utils.py ha sido corregido.")
