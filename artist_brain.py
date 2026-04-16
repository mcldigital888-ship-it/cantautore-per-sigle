"""
Artist Brain — Personalità + Memoria dell'Artista Digitale
===========================================================
Ogni artista ha:
  1. DNA: backstory, personalità, ossessioni, vocabolario, credenze
  2. Memoria: digest delle canzoni precedenti (temi, metafore, frasi chiave)
  3. Evoluzione: l'artista "cresce" canzone dopo canzone

Il DNA viene letto ad ogni generazione e iniettato nel prompt.
La memoria viene aggiornata dopo ogni canzone e usata per coerenza.
"""
import json
from pathlib import Path
from datetime import datetime


class ArtistBrain:
    """Gestisce personalità e memoria di un artista digitale."""

    def __init__(self, artist_id: str, base_dir: Path):
        self.artist_id = artist_id
        self.brain_dir = base_dir / "brain" / artist_id
        self.brain_dir.mkdir(parents=True, exist_ok=True)

        self.dna_file = self.brain_dir / "dna.json"
        self.memory_file = self.brain_dir / "memory.json"
        self.evolution_file = self.brain_dir / "evolution.json"

        # Carica o inizializza
        self.dna = self._load_json(self.dna_file, {})
        self.memory = self._load_json(self.memory_file, {"songs": [], "used_metaphors": [], "used_themes": [], "recurring_words": []})
        self.evolution = self._load_json(self.evolution_file, {"phase": "esordio", "song_count": 0, "artistic_events": []})

    def _load_json(self, path: Path, default: dict) -> dict:
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        return default

    def _save_json(self, path: Path, data: dict):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def save_all(self):
        self._save_json(self.dna_file, self.dna)
        self._save_json(self.memory_file, self.memory)
        self._save_json(self.evolution_file, self.evolution)

    # ============================================================
    # DNA — chi è l'artista
    # ============================================================

    def set_dna(self, dna: dict):
        """Imposta il DNA dell'artista."""
        self.dna = dna
        self._save_json(self.dna_file, self.dna)

    def get_dna_prompt(self) -> str:
        """Genera il blocco di prompt che descrive la personalità."""
        if not self.dna:
            return ""

        d = self.dna
        lines = []
        lines.append(f"CHI SEI (personalità dell'artista — sii fedele a questo in ogni canzone):")
        lines.append(f"Nome: {d.get('name', 'Sconosciuto')}")

        if d.get('backstory'):
            lines.append(f"STORIA: {d['backstory']}")
        if d.get('personality'):
            lines.append(f"PERSONALITÀ: {d['personality']}")
        if d.get('obsessions'):
            lines.append(f"OSSESSIONI RICORRENTI (torna spesso su questi temi): {', '.join(d['obsessions'])}")
        if d.get('vocabulary'):
            lines.append(f"PAROLE CHE AMI (usale naturalmente): {', '.join(d['vocabulary'])}")
        if d.get('forbidden_words'):
            lines.append(f"PAROLE CHE NON USI MAI: {', '.join(d['forbidden_words'])}")
        if d.get('beliefs'):
            lines.append(f"COSA CREDI: {d['beliefs']}")
        if d.get('contradictions'):
            lines.append(f"LE TUE CONTRADDIZIONI (rendono umano): {d['contradictions']}")
        if d.get('places'):
            lines.append(f"I TUOI LUOGHI: {', '.join(d['places'])}")
        if d.get('habits'):
            lines.append(f"ABITUDINI: {d['habits']}")
        if d.get('artistic_voice'):
            lines.append(f"LA TUA VOCE ARTISTICA: {d['artistic_voice']}")

        return "\n".join(lines)

    # ============================================================
    # MEMORIA — cosa hai già scritto
    # ============================================================

    def get_memory_prompt(self) -> str:
        """Genera il blocco di prompt con la memoria delle canzoni precedenti."""
        songs = self.memory.get("songs", [])
        if not songs:
            return "Questa è la tua PRIMA canzone. Stai definendo chi sei. Sii audace."

        lines = []
        lines.append(f"HAI GIÀ SCRITTO {len(songs)} CANZONI. Ricorda:")

        # Ultimi titoli
        titles = [s["titolo"] for s in songs[-10:]]
        lines.append(f"Ultimi titoli: {', '.join(titles)}")

        # Temi già esplorati
        themes = self.memory.get("used_themes", [])
        if themes:
            recent_themes = themes[-15:]
            lines.append(f"Temi già toccati: {', '.join(recent_themes)}")
            lines.append("→ NON ripetere gli stessi temi. Trova un angolo NUOVO.")

        # Metafore già usate
        metaphors = self.memory.get("used_metaphors", [])
        if metaphors:
            recent_metaphors = metaphors[-20:]
            lines.append(f"Metafore già usate (NON ripeterle): {'; '.join(recent_metaphors)}")

        # Parole ricorrenti (queste PUOI riusarle — sono la tua firma)
        recurring = self.memory.get("recurring_words", [])
        if recurring:
            lines.append(f"Parole che tornano nelle tue canzoni (la tua firma): {', '.join(recurring)}")

        # Frasi killer delle canzoni precedenti
        killer_lines = []
        for s in songs[-5:]:
            if s.get("killer_line"):
                killer_lines.append(f'"{s["killer_line"]}" ({s["titolo"]})')
        if killer_lines:
            lines.append(f"Le tue frasi migliori finora: {'; '.join(killer_lines)}")
            lines.append("→ Il livello da mantenere è ALMENO questo. Fai di meglio.")

        return "\n".join(lines)

    def get_evolution_prompt(self) -> str:
        """Genera il blocco che descrive la fase artistica attuale."""
        count = self.evolution.get("song_count", 0)
        phase = self.evolution.get("phase", "esordio")

        if count < 5:
            return f"Sei all'ESORDIO (canzone #{count+1}). Stai trovando la tua voce. Sperimenta, osa, definisci chi sei."
        elif count < 15:
            return f"Sei nella fase di MATURAZIONE (canzone #{count+1}). Hai trovato il tuo stile, ora approfondisci. Sorprendi chi ti segue dall'inizio."
        elif count < 30:
            return f"Sei nella fase di CONSACRAZIONE (canzone #{count+1}). Ogni canzone deve essere un capolavoro. Rischia di più, spacca le convenzioni."
        else:
            return f"Sei un MAESTRO (canzone #{count+1}). Scrivi con la libertà di chi non ha più niente da dimostrare. Vai dove nessuno è mai andato."

    # ============================================================
    # DOPO OGNI CANZONE — aggiorna la memoria
    # ============================================================

    def digest_song(self, song_data: dict, tema: str):
        """Analizza una canzone appena generata e aggiorna la memoria.
        Usa Gemini per estrarre metafore, temi, e la killer line."""
        from google import genai
        from config import GEMINI_API_KEY

        client = genai.Client(api_key=GEMINI_API_KEY)

        prompt = f"""Analizza questo testo di canzone ed estrai in JSON:
{{
    "temi_principali": ["tema1", "tema2", "tema3"],
    "metafore_originali": ["metafora1", "metafora2", "metafora3"],
    "killer_line": "il verso più potente dell'intero testo, quello che da solo vale la canzone",
    "parole_ricorrenti": ["parola1", "parola2"],
    "mood_emotivo": "descrizione breve dell'emozione dominante"
}}

Titolo: {song_data['titolo']}
Tema: {tema}

Testo:
{song_data['testo']}

Rispondi SOLO in JSON."""

        try:
            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt,
            )

            text = response.text.strip()
            if text.startswith("```"):
                text = text.split("\n", 1)[1]
            if text.endswith("```"):
                text = text.rsplit("```", 1)[0]
            text = text.strip()

            analysis = json.loads(text)
        except Exception:
            # Fallback se Gemini fallisce
            analysis = {
                "temi_principali": [tema[:50]],
                "metafore_originali": [],
                "killer_line": "",
                "parole_ricorrenti": [],
                "mood_emotivo": song_data.get("mood", ""),
            }

        # Aggiorna memoria
        song_entry = {
            "titolo": song_data["titolo"],
            "tema": tema,
            "data": datetime.now().isoformat(),
            "mood": song_data.get("mood", ""),
            "bpm": song_data.get("bpm", 0),
            "killer_line": analysis.get("killer_line", ""),
            "mood_emotivo": analysis.get("mood_emotivo", ""),
        }
        self.memory["songs"].append(song_entry)

        # Accumula temi, metafore, parole
        for t in analysis.get("temi_principali", []):
            if t not in self.memory["used_themes"]:
                self.memory["used_themes"].append(t)

        for m in analysis.get("metafore_originali", []):
            if m not in self.memory["used_metaphors"]:
                self.memory["used_metaphors"].append(m)

        for w in analysis.get("parole_ricorrenti", []):
            if w not in self.memory["recurring_words"]:
                self.memory["recurring_words"].append(w)

        # Aggiorna evoluzione
        self.evolution["song_count"] = len(self.memory["songs"])

        self.save_all()
        return analysis
