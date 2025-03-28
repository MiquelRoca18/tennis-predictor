import pandas as pd
import numpy as np
from datetime import datetime
import psycopg2
from psycopg2.extras import execute_batch

class EnhancedTennisELO:
    """
    Sistema ELO mejorado para tenis que implementa:
    - ELO específico por superficie
    - Factores K dinámicos basados en importancia del torneo/ronda
    - Ponderación por margen de victoria
    - Decaimiento temporal
    """
    
    def __init__(self, db_connection_string):
        """
        Inicializa el sistema ELO con conexión a la base de datos
        
        Args:
            db_connection_string: String de conexión a PostgreSQL
        """
        self.conn_string = db_connection_string
        self.default_elo = 1500
        self.decay_rate = 0.995  # Factor de decaimiento por inactividad (mensual)
        
        # Factor K base por tipo de torneo
        self.tournament_k_factors = {
            'G': 40,  # Grand Slam
            'M': 30,  # Masters 1000
            'A': 25,  # ATP 500
            'D': 20,  # ATP 250
            'F': 35,  # Finals
            'O': 15   # Otros
        }
        
        # Multiplicadores por ronda (aumenta en rondas finales)
        self.round_multipliers = {
            'F': 1.5,     # Final
            'SF': 1.3,    # Semifinal
            'QF': 1.2,    # Cuartos de final
            'R16': 1.1,   # Octavos
            'R32': 1.0,   # 1/16
            'R64': 0.9,   # 1/32
            'R128': 0.8   # 1/64
        }
        
        # Multiplicadores por superficie
        self.surface_specificity = {
            'Hard': 1.0,
            'Clay': 1.1,   # Mayor especificidad en tierra
            'Grass': 1.2,  # Mayor especificidad en hierba
            'Carpet': 1.1
        }
    
    def _get_connection(self):
        """Establece conexión con la base de datos"""
        return psycopg2.connect(self.conn_string)
    
    def setup_tables(self):
        """Configura las tablas necesarias para el sistema ELO"""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        # Añadir columnas ELO a la tabla de jugadores
        try:
            cursor.execute("""
            ALTER TABLE players 
            ADD COLUMN IF NOT EXISTS elo_rating FLOAT DEFAULT 1500,
            ADD COLUMN IF NOT EXISTS elo_hard FLOAT DEFAULT 1500,
            ADD COLUMN IF NOT EXISTS elo_clay FLOAT DEFAULT 1500,
            ADD COLUMN IF NOT EXISTS elo_grass FLOAT DEFAULT 1500,
            ADD COLUMN IF NOT EXISTS elo_carpet FLOAT DEFAULT 1500,
            ADD COLUMN IF NOT EXISTS elo_matches_played INT DEFAULT 0,
            ADD COLUMN IF NOT EXISTS elo_last_update DATE;
            """)
            
            # Crear tabla de historial ELO
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS player_elo_history (
                id SERIAL PRIMARY KEY,
                player_id INTEGER REFERENCES players(id),
                match_id INTEGER REFERENCES matches(id),
                date DATE NOT NULL,
                elo_rating FLOAT NOT NULL,
                elo_hard FLOAT,
                elo_clay FLOAT,
                elo_grass FLOAT,
                elo_carpet FLOAT,
                elo_change FLOAT
            );
            """)
            
            conn.commit()
            print("Tablas ELO configuradas correctamente")
        except Exception as e:
            conn.rollback()
            print(f"Error configurando tablas: {e}")
        finally:
            cursor.close()
            conn.close()
    
    def calculate_expected_win_probability(self, player_elo, opponent_elo):
        """
        Calcula la probabilidad esperada de victoria
        
        Args:
            player_elo: Rating ELO del jugador
            opponent_elo: Rating ELO del oponente
            
        Returns:
            Probabilidad esperada de victoria (0-1)
        """
        return 1.0 / (1.0 + 10**((opponent_elo - player_elo) / 400.0))
    
    def _get_k_factor(self, player_matches, tournament_type, round_name, surface):
        """
        Determina el factor K dinámico basado en múltiples factores
        
        Args:
            player_matches: Número de partidos jugados por el jugador
            tournament_type: Tipo de torneo (G, M, A, D, F, O)
            round_name: Ronda del torneo (F, SF, QF, etc.)
            surface: Superficie de juego
            
        Returns:
            Factor K ajustado
        """
        # Factor base por tipo de torneo
        base_k = self.tournament_k_factors.get(tournament_type, 20)
        
        # Ajuste por ronda
        round_mult = self.round_multipliers.get(round_name, 1.0)
        
        # Ajuste por superficie
        surface_mult = self.surface_specificity.get(surface, 1.0)
        
        # Ajuste por experiencia (menor K para jugadores con más experiencia)
        experience_mult = max(0.7, 1.0 - (player_matches / 500))
        
        # Calcular K final
        k_factor = base_k * round_mult * surface_mult * experience_mult
        
        return k_factor
    
    def _get_margin_multiplier(self, score, retirement=False):
        """
        Calcula multiplicador basado en el margen de victoria
        
        Args:
            score: String con el resultado (e.g. '6-4 7-5')
            retirement: Indica si hubo abandono
            
        Returns:
            Multiplicador basado en el margen de victoria
        """
        if retirement:
            return 0.8  # Menor impacto para partidos con abandono
            
        try:
            # Analizar el marcador
            sets = score.split()
            sets_won_winner = 0
            sets_won_loser = 0
            games_diff = 0
            
            for set_score in sets:
                if '-' in set_score:
                    games = set_score.split('-')
                    if len(games) == 2:
                        winner_games = int(games[0])
                        loser_games = int(games[1])
                        
                        games_diff += winner_games - loser_games
                        
                        if winner_games > loser_games:
                            sets_won_winner += 1
                        else:
                            sets_won_loser += 1
            
            # Calcular el multiplicador basado en dominio del partido
            sets_diff = sets_won_winner - sets_won_loser
            dominance = (games_diff / max(len(sets), 1)) / 6.0  # Normalizado
            
            # Valor entre 0.8 y 1.2 según el dominio
            return 0.8 + (dominance * 0.4)
            
        except Exception:
            # Si hay error al analizar, usar valor por defecto
            return 1.0
    
    def update_elo_for_match(self, winner_id, loser_id, surface, tournament_type, 
                            round_name, score, match_date, retirement=False):
        """
        Actualiza el ELO para un partido individual
        
        Args:
            winner_id: ID del jugador ganador
            loser_id: ID del jugador perdedor
            surface: Superficie de juego
            tournament_type: Tipo de torneo
            round_name: Ronda del torneo
            score: Marcador del partido
            match_date: Fecha del partido
            retirement: Indica si hubo abandono
            
        Returns:
            Tupla con (cambio_elo_ganador, cambio_elo_perdedor)
        """
        conn = self._get_connection()
        cursor = conn.cursor()
        
        try:
            # Obtener datos actuales de los jugadores
            cursor.execute("""
            SELECT id, elo_rating, elo_hard, elo_clay, elo_grass, elo_carpet, 
                   elo_matches_played, elo_last_update 
            FROM players 
            WHERE id IN (%s, %s)
            """, (winner_id, loser_id))
            
            players_data = {}
            for row in cursor.fetchall():
                players_data[row[0]] = {
                    'elo': row[1],
                    f'elo_{surface.lower()}': row[2 + ['hard', 'clay', 'grass', 'carpet'].index(surface.lower())],
                    'matches': row[6],
                    'last_update': row[7]
                }
            
            # Si algún jugador no existe, crear entrada con valores por defecto
            if winner_id not in players_data:
                players_data[winner_id] = {
                    'elo': self.default_elo,
                    f'elo_{surface.lower()}': self.default_elo,
                    'matches': 0,
                    'last_update': match_date
                }
            
            if loser_id not in players_data:
                players_data[loser_id] = {
                    'elo': self.default_elo,
                    f'elo_{surface.lower()}': self.default_elo,
                    'matches': 0,
                    'last_update': match_date
                }
            
            # Calcular decaimiento por inactividad si corresponde
            for player_id in [winner_id, loser_id]:
                last_update = players_data[player_id]['last_update']
                if last_update:
                    months_inactive = (match_date - last_update).days / 30.0
                    if months_inactive > 1:
                        decay = self.decay_rate ** months_inactive
                        players_data[player_id]['elo'] *= decay
                        players_data[player_id][f'elo_{surface.lower()}'] *= decay
            
            # Calcular probabilidades esperadas (general y por superficie)
            w_general_elo = players_data[winner_id]['elo']
            l_general_elo = players_data[loser_id]['elo']
            w_surface_elo = players_data[winner_id][f'elo_{surface.lower()}']
            l_surface_elo = players_data[loser_id][f'elo_{surface.lower()}']
            
            # Probabilidad esperada (media ponderada entre general y superficie)
            w_win_prob_general = self.calculate_expected_win_probability(w_general_elo, l_general_elo)
            w_win_prob_surface = self.calculate_expected_win_probability(w_surface_elo, l_surface_elo)
            
            # 70% superficie específica, 30% general
            w_win_prob = 0.3 * w_win_prob_general + 0.7 * w_win_prob_surface
            
            # Determinar factores K
            k_winner = self._get_k_factor(
                players_data[winner_id]['matches'],
                tournament_type,
                round_name,
                surface
            )
            
            k_loser = self._get_k_factor(
                players_data[loser_id]['matches'],
                tournament_type,
                round_name,
                surface
            )
            
            # Ajustar K por margen de victoria
            margin_multiplier = self._get_margin_multiplier(score, retirement)
            
            # Calcular cambios de ELO
            elo_change_winner = k_winner * margin_multiplier * (1 - w_win_prob)
            elo_change_loser = k_loser * margin_multiplier * (0 - (1 - w_win_prob))
            
            # Actualizar ELO general
            players_data[winner_id]['elo'] += elo_change_winner
            players_data[loser_id]['elo'] += elo_change_loser
            
            # Actualizar ELO por superficie (cambio mayor en la superficie específica)
            surface_mult = self.surface_specificity.get(surface, 1.0)
            players_data[winner_id][f'elo_{surface.lower()}'] += elo_change_winner * surface_mult
            players_data[loser_id][f'elo_{surface.lower()}'] += elo_change_loser * surface_mult
            
            # Actualizar contador de partidos
            players_data[winner_id]['matches'] += 1
            players_data[loser_id]['matches'] += 1
            
            # Actualizar datos en la base de datos
            for player_id in [winner_id, loser_id]:
                cursor.execute("""
                UPDATE players 
                SET elo_rating = %s,
                    elo_hard = CASE WHEN %s = 'hard' THEN %s ELSE elo_hard END,
                    elo_clay = CASE WHEN %s = 'clay' THEN %s ELSE elo_clay END,
                    elo_grass = CASE WHEN %s = 'grass' THEN %s ELSE elo_grass END,
                    elo_carpet = CASE WHEN %s = 'carpet' THEN %s ELSE elo_carpet END,
                    elo_matches_played = %s,
                    elo_last_update = %s
                WHERE id = %s
                """, (
                    players_data[player_id]['elo'],
                    surface.lower(), players_data[player_id][f'elo_{surface.lower()}'],
                    surface.lower(), players_data[player_id][f'elo_{surface.lower()}'],
                    surface.lower(), players_data[player_id][f'elo_{surface.lower()}'],
                    surface.lower(), players_data[player_id][f'elo_{surface.lower()}'],
                    players_data[player_id]['matches'],
                    match_date,
                    player_id
                ))
            
            conn.commit()
            return elo_change_winner, elo_change_loser
            
        except Exception as e:
            conn.rollback()
            print(f"Error actualizando ELO: {e}")
            return 0, 0
        finally:
            cursor.close()
            conn.close()
    
    def recalculate_historical_elo(self, start_date=None, end_date=None):
        """
        Recalcula todo el historial de ELO desde cero
        
        Args:
            start_date: Fecha de inicio para el recálculo (opcional)
            end_date: Fecha final para el recálculo (opcional)
        """
        conn = self._get_connection()
        cursor = conn.cursor()
        
        try:
            # Reiniciar ELO para todos los jugadores
            cursor.execute("""
            UPDATE players
            SET elo_rating = 1500,
                elo_hard = 1500,
                elo_clay = 1500,
                elo_grass = 1500,
                elo_carpet = 1500,
                elo_matches_played = 0,
                elo_last_update = NULL;
            """)
            
            # Limpiar historial previo
            cursor.execute("TRUNCATE TABLE player_elo_history;")
            
            # Construir condición de fecha
            date_condition = ""
            params = []
            
            if start_date:
                date_condition += " AND date >= %s"
                params.append(start_date)
            
            if end_date:
                date_condition += " AND date <= %s"
                params.append(end_date)
            
            # Obtener todos los partidos ordenados por fecha
            cursor.execute(f"""
            SELECT id, winner_id, loser_id, 
                   LOWER(surface) as surface, 
                   tourney_level as tournament_type,
                   UPPER(round) as round_name,
                   score,
                   date,
                   retirement
            FROM matches
            WHERE winner_id IS NOT NULL AND loser_id IS NOT NULL
            {date_condition}
            ORDER BY date ASC, id ASC;
            """, params)
            
            matches = cursor.fetchall()
            total_matches = len(matches)
            print(f"Procesando {total_matches} partidos...")
            
            # Procesar cada partido
            elo_history_records = []
            batch_size = 1000
            processed = 0
            
            for match in matches:
                match_id, winner_id, loser_id, surface, tournament_type, round_name, score, match_date, retirement = match
                
                # Si la superficie no es válida, asignar 'hard' por defecto
                if surface not in ['hard', 'clay', 'grass', 'carpet']:
                    surface = 'hard'
                
                # Actualizar ELO
                elo_change_winner, elo_change_loser = self.update_elo_for_match(
                    winner_id, loser_id, surface, tournament_type, 
                    round_name, score, match_date, retirement
                )
                
                # Obtener ELO actualizado
                cursor.execute("""
                SELECT id, elo_rating, elo_hard, elo_clay, elo_grass, elo_carpet
                FROM players 
                WHERE id IN (%s, %s)
                """, (winner_id, loser_id))
                
                player_elos = {}
                for row in cursor.fetchall():
                    player_elos[row[0]] = (row[1], row[2], row[3], row[4], row[5])
                
                # Registrar historial
                if winner_id in player_elos:
                    elo_history_records.append((
                        winner_id, match_id, match_date, 
                        player_elos[winner_id][0],  # elo_rating
                        player_elos[winner_id][1],  # elo_hard
                        player_elos[winner_id][2],  # elo_clay
                        player_elos[winner_id][3],  # elo_grass
                        player_elos[winner_id][4],  # elo_carpet
                        elo_change_winner
                    ))
                
                if loser_id in player_elos:
                    elo_history_records.append((
                        loser_id, match_id, match_date, 
                        player_elos[loser_id][0],   # elo_rating
                        player_elos[loser_id][1],   # elo_hard
                        player_elos[loser_id][2],   # elo_clay
                        player_elos[loser_id][3],   # elo_grass
                        player_elos[loser_id][4],   # elo_carpet
                        elo_change_loser
                    ))
                
                # Insertar en lotes para mejor rendimiento
                if len(elo_history_records) >= batch_size:
                    execute_batch(cursor, """
                    INSERT INTO player_elo_history
                    (player_id, match_id, date, elo_rating, elo_hard, elo_clay, elo_grass, elo_carpet, elo_change)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                    """, elo_history_records)
                    elo_history_records = []
                    conn.commit()
                
                processed += 1
                if processed % 1000 == 0:
                    print(f"Procesados {processed}/{total_matches} partidos...")
            
            # Insertar registros restantes
            if elo_history_records:
                execute_batch(cursor, """
                INSERT INTO player_elo_history
                (player_id, match_id, date, elo_rating, elo_hard, elo_clay, elo_grass, elo_carpet, elo_change)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                """, elo_history_records)
            
            conn.commit()
            print(f"Recálculo de ELO completado para {processed} partidos")
            
        except Exception as e:
            conn.rollback()
            print(f"Error en recálculo de ELO: {e}")
            raise
        finally:
            cursor.close()
            conn.close()
    
    def get_player_elo(self, player_id, date=None, surface=None):
        """
        Obtiene el rating ELO de un jugador en una fecha específica
        
        Args:
            player_id: ID del jugador
            date: Fecha para la cual obtener el ELO (por defecto actual)
            surface: Superficie específica (opcional)
            
        Returns:
            Rating ELO o tupla (general, específico) si se especifica superficie
        """
        conn = self._get_connection()
        cursor = conn.cursor()
        
        try:
            if date:
                # Obtener ELO histórico más cercano a la fecha
                surface_column = f"elo_{surface.lower()}" if surface else "elo_rating"
                
                cursor.execute(f"""
                SELECT elo_rating, {surface_column}
                FROM player_elo_history
                WHERE player_id = %s AND date <= %s
                ORDER BY date DESC, id DESC
                LIMIT 1
                """, (player_id, date))
                
                result = cursor.fetchone()
                if result:
                    return result[1] if surface else result[0]
                else:
                    return self.default_elo
            else:
                # Obtener ELO actual
                cursor.execute("""
                SELECT elo_rating, elo_hard, elo_clay, elo_grass, elo_carpet
                FROM players
                WHERE id = %s
                """, (player_id,))
                
                result = cursor.fetchone()
                if result:
                    if surface:
                        surface_index = ['hard', 'clay', 'grass', 'carpet'].index(surface.lower())
                        return (result[0], result[1 + surface_index])
                    else:
                        return result[0]
                else:
                    return self.default_elo
        
        except Exception as e:
            print(f"Error obteniendo ELO: {e}")
            return self.default_elo
        finally:
            cursor.close()
            conn.close()
    
    def create_elo_features(self, matches_df):
        """
        Agrega características de ELO a un DataFrame de partidos
        
        Args:
            matches_df: DataFrame con partidos (debe tener winner_id, loser_id, date, surface)
            
        Returns:
            DataFrame con características ELO añadidas
        """
        # Copiar DataFrame para no modificar el original
        df = matches_df.copy()
        
        # Inicializar columnas
        df['player1_elo'] = 1500
        df['player2_elo'] = 1500
        df['player1_surface_elo'] = 1500
        df['player2_surface_elo'] = 1500
        df['elo_difference'] = 0
        df['surface_elo_difference'] = 0
        
        # Conexión a la base de datos
        conn = self._get_connection()
        cursor = conn.cursor()
        
        try:
            for idx, row in df.iterrows():
                # Asignar player1 y player2 según corresponda
                p1_id = row['winner_id'] if 'winner_id' in df.columns else row['player1_id']
                p2_id = row['loser_id'] if 'loser_id' in df.columns else row['player2_id']
                match_date = row['date']
                surface = row['surface'].lower() if 'surface' in df.columns else 'hard'
                
                # Normalizar superficie
                if surface not in ['hard', 'clay', 'grass', 'carpet']:
                    surface = 'hard'
                
                # Obtener ELO para player1
                cursor.execute("""
                SELECT elo_rating, elo_hard, elo_clay, elo_grass, elo_carpet
                FROM player_elo_history
                WHERE player_id = %s AND date < %s
                ORDER BY date DESC, id DESC
                LIMIT 1
                """, (p1_id, match_date))
                
                p1_elo = cursor.fetchone()
                if p1_elo:
                    df.at[idx, 'player1_elo'] = p1_elo[0]  # ELO general
                    surface_idx = ['hard', 'clay', 'grass', 'carpet'].index(surface)
                    df.at[idx, 'player1_surface_elo'] = p1_elo[1 + surface_idx]  # ELO específico
                else:
                    # Usar valor por defecto si no hay historial
                    df.at[idx, 'player1_elo'] = self.default_elo
                    df.at[idx, 'player1_surface_elo'] = self.default_elo
                
                # Obtener ELO para player2
                cursor.execute("""
                SELECT elo_rating, elo_hard, elo_clay, elo_grass, elo_carpet
                FROM player_elo_history
                WHERE player_id = %s AND date < %s
                ORDER BY date DESC, id DESC
                LIMIT 1
                """, (p2_id, match_date))
                
                p2_elo = cursor.fetchone()
                if p2_elo:
                    df.at[idx, 'player2_elo'] = p2_elo[0]  # ELO general
                    surface_idx = ['hard', 'clay', 'grass', 'carpet'].index(surface)
                    df.at[idx, 'player2_surface_elo'] = p2_elo[1 + surface_idx]  # ELO específico
                else:
                    # Usar valor por defecto si no hay historial
                    df.at[idx, 'player2_elo'] = self.default_elo
                    df.at[idx, 'player2_surface_elo'] = self.default_elo
                
                # Calcular diferencias
                df.at[idx, 'elo_difference'] = df.at[idx, 'player1_elo'] - df.at[idx, 'player2_elo']
                df.at[idx, 'surface_elo_difference'] = df.at[idx, 'player1_surface_elo'] - df.at[idx, 'player2_surface_elo']
            
            return df
        
        except Exception as e:
            print(f"Error creando características ELO: {e}")
            return df
        finally:
            cursor.close()
            conn.close()
    
    def create_postgresql_trigger(self):
        """
        Crea un trigger en PostgreSQL para actualizar ELO automáticamente al insertar partidos
        """
        conn = self._get_connection()
        cursor = conn.cursor()
        
        try:
            # Crear función para el trigger
            cursor.execute("""
            CREATE OR REPLACE FUNCTION update_elo_after_match()
            RETURNS TRIGGER AS $
            DECLARE
                w_elo FLOAT;
                l_elo FLOAT;
                w_surface_elo FLOAT;
                l_surface_elo FLOAT;
                exp_win FLOAT;
                k_factor FLOAT;
                elo_change FLOAT;
                surface_field TEXT;
                tournament_k FLOAT;
                round_mult FLOAT;
                surface_mult FLOAT;
                w_matches INT;
                l_matches INT;
            BEGIN
                -- Obtener ELO actual
                SELECT elo_rating, elo_matches_played INTO w_elo, w_matches FROM players WHERE id = NEW.winner_id;
                SELECT elo_rating, elo_matches_played INTO l_elo, l_matches FROM players WHERE id = NEW.loser_id;
                
                -- Valores por defecto si no existen
                IF w_elo IS NULL THEN w_elo := 1500; w_matches := 0; END IF;
                IF l_elo IS NULL THEN l_elo := 1500; l_matches := 0; END IF;
                
                -- Determinar campo de superficie
                CASE LOWER(NEW.surface)
                    WHEN 'hard' THEN surface_field := 'elo_hard';
                    WHEN 'clay' THEN surface_field := 'elo_clay';
                    WHEN 'grass' THEN surface_field := 'elo_grass';
                    WHEN 'carpet' THEN surface_field := 'elo_carpet';
                    ELSE surface_field := 'elo_hard';
                END CASE;
                
                -- Obtener ELO específico por superficie
                EXECUTE 'SELECT ' || surface_field || ' FROM players WHERE id = $1'
                INTO w_surface_elo
                USING NEW.winner_id;
                
                EXECUTE 'SELECT ' || surface_field || ' FROM players WHERE id = $1'
                INTO l_surface_elo
                USING NEW.loser_id;
                
                -- Valores por defecto si no existen
                IF w_surface_elo IS NULL THEN w_surface_elo := 1500; END IF;
                IF l_surface_elo IS NULL THEN l_surface_elo := 1500; END IF;
                
                -- Calcular probabilidad esperada (70% superficie, 30% general)
                exp_win := 0.3 * (1 / (1 + 10^((l_elo - w_elo)/400))) + 
                           0.7 * (1 / (1 + 10^((l_surface_elo - w_surface_elo)/400)));
                
                -- Determinar factor K
                CASE NEW.tourney_level
                    WHEN 'G' THEN tournament_k := 40;
                    WHEN 'M' THEN tournament_k := 30;
                    WHEN 'A' THEN tournament_k := 25;
                    WHEN 'D' THEN tournament_k := 20;
                    WHEN 'F' THEN tournament_k := 35;
                    ELSE tournament_k := 15;
                END CASE;
                
                -- Multiplicador por ronda
                CASE UPPER(NEW.round)
                    WHEN 'F' THEN round_mult := 1.5;
                    WHEN 'SF' THEN round_mult := 1.3;
                    WHEN 'QF' THEN round_mult := 1.2;
                    WHEN 'R16' THEN round_mult := 1.1;
                    WHEN 'R32' THEN round_mult := 1.0;
                    WHEN 'R64' THEN round_mult := 0.9;
                    WHEN 'R128' THEN round_mult := 0.8;
                    ELSE round_mult := 1.0;
                END CASE;
                
                -- Multiplicador por superficie
                CASE LOWER(NEW.surface)
                    WHEN 'hard' THEN surface_mult := 1.0;
                    WHEN 'clay' THEN surface_mult := 1.1;
                    WHEN 'grass' THEN surface_mult := 1.2;
                    WHEN 'carpet' THEN surface_mult := 1.1;
                    ELSE surface_mult := 1.0;
                END CASE;
                
                -- Calcular K final para ganador
                k_factor := tournament_k * round_mult * 
                            MAX(0.7, 1.0 - (w_matches / 500));
                
                -- Cambio de ELO (ganador)
                elo_change := k_factor * (1 - exp_win);
                
                -- Actualizar ELO general del ganador
                UPDATE players 
                SET elo_rating = elo_rating + elo_change,
                    elo_matches_played = elo_matches_played + 1,
                    elo_last_update = NEW.date
                WHERE id = NEW.winner_id;
                
                -- Actualizar ELO específico por superficie del ganador
                EXECUTE 'UPDATE players SET ' || surface_field || ' = ' || surface_field || 
                        ' + $1 * $2 WHERE id = $3'
                USING elo_change, surface_mult, NEW.winner_id;
                
                -- Calcular K final para perdedor
                k_factor := tournament_k * round_mult * 
                            MAX(0.7, 1.0 - (l_matches / 500));
                
                -- Cambio de ELO (perdedor)
                elo_change := k_factor * (0 - (1 - exp_win));
                
                -- Actualizar ELO general del perdedor
                UPDATE players 
                SET elo_rating = elo_rating + elo_change,
                    elo_matches_played = elo_matches_played + 1,
                    elo_last_update = NEW.date
                WHERE id = NEW.loser_id;
                
                -- Actualizar ELO específico por superficie del perdedor
                EXECUTE 'UPDATE players SET ' || surface_field || ' = ' || surface_field || 
                        ' + $1 * $2 WHERE id = $3'
                USING elo_change, surface_mult, NEW.loser_id;
                
                -- Guardar historial
                INSERT INTO player_elo_history 
                (player_id, match_id, date, elo_rating, elo_change)
                SELECT NEW.winner_id, NEW.id, NEW.date, elo_rating, ABS(elo_change)
                FROM players WHERE id = NEW.winner_id;
                
                INSERT INTO player_elo_history 
                (player_id, match_id, date, elo_rating, elo_change)
                SELECT NEW.loser_id, NEW.id, NEW.date, elo_rating, elo_change
                FROM players WHERE id = NEW.loser_id;
                
                RETURN NEW;
            END;
            $ LANGUAGE plpgsql;
            """)
            
            # Crear trigger
            cursor.execute("""
            DROP TRIGGER IF EXISTS update_elo_trigger ON matches;
            CREATE TRIGGER update_elo_trigger
            AFTER INSERT ON matches
            FOR EACH ROW
            EXECUTE FUNCTION update_elo_after_match();
            """)
            
            conn.commit()
            print("Trigger de actualización ELO creado exitosamente")
        except Exception as e:
            conn.rollback()
            print(f"Error creando trigger: {e}")
        finally:
            cursor.close()
            conn.close()
    
    def evaluate_elo_predictive_power(self, test_matches_df):
        """
        Evalúa el poder predictivo del sistema ELO
        
        Args:
            test_matches_df: DataFrame con partidos de prueba
            
        Returns:
            Dict con métricas de rendimiento
        """
        # Añadir características ELO
        df = self.create_elo_features(test_matches_df)
        
        # Crear columna para predicción basada en ELO
        df['predicted_winner'] = (df['player1_surface_elo'] > df['player2_surface_elo']).astype(int)
        
        # La columna target debe ser 1 si player1 ganó, 0 si player2 ganó
        if 'winner_id' in df.columns and 'player1_id' in df.columns:
            df['actual_winner'] = (df['winner_id'] == df['player1_id']).astype(int)
        else:
            # Si el dataset ya tiene la columna de resultado
            df['actual_winner'] = df['target'] if 'target' in df.columns else 1
        
        # Calcular métricas
        correct_predictions = (df['predicted_winner'] == df['actual_winner']).sum()
        total_matches = len(df)
        accuracy = correct_predictions / total_matches if total_matches > 0 else 0
        
        # Calcular precisión por superficie
        surface_accuracy = {}
        for surface in df['surface'].unique():
            surface_df = df[df['surface'] == surface]
            if len(surface_df) > 0:
                surface_correct = (surface_df['predicted_winner'] == surface_df['actual_winner']).sum()
                surface_accuracy[surface] = surface_correct / len(surface_df)
        
        # Calcular precisión por diferencia de ELO
        df['elo_diff_abs'] = abs(df['surface_elo_difference'])
        df['elo_diff_bin'] = pd.cut(df['elo_diff_abs'], 
                                    bins=[0, 50, 100, 150, 200, float('inf')],
                                    labels=['0-50', '50-100', '100-150', '150-200', '200+'])
        
        elo_diff_accuracy = {}
        for bin_name in df['elo_diff_bin'].unique():
            if pd.isna(bin_name):
                continue
            bin_df = df[df['elo_diff_bin'] == bin_name]
            if len(bin_df) > 0:
                bin_correct = (bin_df['predicted_winner'] == bin_df['actual_winner']).sum()
                elo_diff_accuracy[bin_name] = bin_correct / len(bin_df)
        
        return {
            'accuracy': accuracy,
            'surface_accuracy': surface_accuracy,
            'elo_diff_accuracy': elo_diff_accuracy,
            'total_matches': total_matches,
            'correct_predictions': correct_predictions
        }