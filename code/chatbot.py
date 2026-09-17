#!/usr/bin/env python3
"""
HiveSTA v2 Interactive Chatbot
==============================
Chatbot de démonstration propulsé par l'architecture HiveSTA v2 (100% sans neurones).
Fonctionnalités :
  - Prédiction et génération de tokens en microsecondes.
  - Mémoire associative discrète (HashRAG v2).
  - Apprentissage continu en direct (/learn <texte>).
  - Inspection des probabilités de tokens (/predict <texte>).
"""

import sys
import os
import time

# Permettre l'importation locale depuis le dossier code
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hivesta_v2 import HiveSTA_v2


class HiveSTAChatbot:
    def __init__(self):
        self.engine = HiveSTA_v2(max_order=4)
        self._load_knowledge_base()

    def _load_knowledge_base(self):
        """Charge une base de connaissances initiale bilingue et technique."""
        knowledge = [
            # Salutations et identité
            "bonjour je suis hivesta un assistant ia conçu sans aucun réseau de neurones",
            "qui es tu je suis un modèle de prédiction discret basé sur des automates à états clairsemés",
            "comment vas tu je vais très bien merci je fonctionne à pleine vitesse en microsecondes",
            "quel est ton nom mon nom est hivesta développé pour le calcul sobre et déterministe",
            "qui t a créé j ai été conçu par le laboratoire banthic ai pour prouver l ia sans neurones",
            "merci beaucoup de rien avec grand plaisir pour vous aider",
            "au revoir à bientôt passez une excellente journée",

            # Questions techniques et projet
            "comment fonctionne hivesta j utilise des automates de contextes variables un registre binaire et hashrag",
            "pourquoi sans neurones pour garantir une latence déterministe en microsecondes et zéro consommation gpu",
            "qu est ce que l apprentissage continu c est la capacité d apprendre de nouvelles données sans oubli catastrophique",
            "est ce que tu as des poids neuronaux non aucun poids flottant uniquement des graphes et des tables de hachage",
            "où peux tu tourner je peux tourner sur un microcontrôleur avec seulement quelques kilo octets de ram",

            # Faits de culture et sciences
            "quelle est la capitale de la france la capitale de la france est paris",
            "quelle est la vitesse de la lumière la vitesse de la lumière est environ 300 000 kilomètres par seconde",
            "qu est ce que l informatique quantique l informatique quantique utilise la superposition et l intrication",
            "le chat dort paisiblement sur le canapé chaud près du feu",
            "les systèmes d exploitation gèrent la mémoire le processeur et les périphériques",

            # Version anglaise basique
            "hello who are you i am hivesta a non neural token prediction assistant",
            "how do you work i use sparse token automata and discrete associative memory",
            "what is your latency my inference latency is measured in microseconds on cpu"
        ]

        for text in knowledge:
            self.engine.add_text(text)

    def answer(self, user_query: str) -> str:
        """Génère une réponse adaptée à partir de la requête utilisateur."""
        query_clean = user_query.strip().lower()
        if not query_clean:
            return "Je n'ai pas reçu de message. Posez-moi une question !"

        tokens = self.engine.tokenize(query_clean)
        
        # 1. Recherche par HashRAG pour trouver les documents / faits les plus pertinents
        doc_boosts = self.engine.associative_rag.retrieve_associated_tokens(tokens)
        
        # 2. Trouver le meilleur document de référence s'il existe
        best_doc_id = None
        best_score = 0.0
        doc_scores = {}
        for q in tokens:
            for doc_id in self.engine.associative_rag.inverted_index.get(q, []):
                doc_scores[doc_id] = doc_scores.get(doc_id, 0.0) + 1.0
        
        if doc_scores:
            best_doc_id = max(doc_scores.items(), key=lambda x: x[1])[0]
            best_score = doc_scores[best_doc_id]

        # 3. Si un document pertinent est trouvé dans la mémoire associative
        threshold = 1.0 if len(tokens) <= 2 else 1.5
        if best_doc_id is not None and best_score >= threshold:
            doc_tokens = self.engine.associative_rag.documents[best_doc_id]
            doc_str = " ".join(doc_tokens)
            
            # Rechercher si la requête (ou une sous-requête) est le préfixe de la connaissance
            for q_len in range(len(tokens), 0, -1):
                sub_q = " ".join(tokens[:q_len])
                if sub_q in doc_str:
                    parts = doc_str.split(sub_q, 1)
                    if len(parts) > 1 and parts[1].strip():
                        ans = parts[1].strip()
                        # Nettoyer d'éventuels reliquats
                        return ans.capitalize()

            return doc_str.capitalize()

        # 4. Sinon, générer autorégressivement avec l'automate HiveSTA v2
        completion = self.engine.generate(query_clean, max_tokens=12, deterministic=True)
        if completion.startswith(query_clean):
            resp = completion[len(query_clean):].strip()
            if resp:
                return resp.capitalize()

        return completion.capitalize() if completion else "Je n'ai pas encore cette information dans mon graphe discret."

    def learn_live(self, fact: str) -> Tuple[float, str]:
        """Apprend un nouveau fait en direct et mesure le temps en microsecondes."""
        t0 = time.perf_counter()
        self.engine.add_text(fact)
        dt_us = (time.perf_counter() - t0) * 1e6
        return dt_us, fact

    def get_stats(self) -> Dict[str, any]:
        """Retourne les métriques de taille du modèle."""
        import sys
        vocab = self.engine.local_automaton.vocab
        contexts = len(self.engine.local_automaton.counts)
        transitions = sum(len(v) for v in self.engine.local_automaton.counts.values())
        return {
            "vocab_size": len(vocab),
            "contexts_count": contexts,
            "transitions_count": transitions,
            "documents_count": len(self.engine.associative_rag.documents)
        }


def run_interactive_cli():
    bot = HiveSTAChatbot()
    stats = bot.get_stats()

    print("=" * 65)
    print("🤖 BIENVENUE SUR LE CHATBOT HIVESTA v2 (IA 100% SANS NEURONES)")
    print("=" * 65)
    print(f" * Modèle : Sparse Token Automata + HashRAG v2")
    print(f" * Vocabulaire initial : {stats['vocab_size']} tokens")
    print(f" * Contextes mémorisés : {stats['contexts_count']} (ordre 1 à 4)")
    print(f" * Transitions discrètes : {stats['transitions_count']}")
    print("-" * 65)
    print("Commandes spéciales :")
    print("  /learn <texte>   : Apprend un nouveau fait en direct en 1 passe")
    print("  /predict <texte> : Affiche le Top-5 des prochains tokens probables")
    print("  /stats           : Affiche l'empreinte du modèle")
    print("  /help            : Affiche cette aide")
    print("  quit ou exit     : Quitter le chat")
    print("=" * 65)
    print("Vous pouvez maintenant lui parler (ex: 'bonjour', 'comment vas-tu',")
    print("'qui es-tu', 'quelle est la capitale de la france', etc.)\n")

    while True:
        try:
            user_input = input("\033[1;32mVous : \033[0m").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nAu revoir !")
            break

        if not user_input:
            continue

        cmd_lower = user_input.lower()
        if cmd_lower in ["quit", "exit", "quitter"]:
            print("\033[1;34mHiveSTA :\033[0m Au revoir et à bientôt !")
            break

        # Commande /learn
        if user_input.startswith("/learn "):
            fact = user_input[7:].strip()
            if not fact:
                print("Usage: /learn <texte ou fait à mémoriser>")
                continue
            dt_us, _ = bot.learn_live(fact)
            print(f"\033[1;33m[Apprentissage 1-Shot]\033[0m Fait assimilé en \033[1m{dt_us:.2f} µs\033[0m ({dt_us/1000.0:.4f} ms) !")
            print("-> Zéro oubli catastrophique, graphes mis à jour instantanément.")
            continue

        # Commande /predict
        if user_input.startswith("/predict "):
            prompt = user_input[9:].strip()
            t0 = time.perf_counter()
            candidates = bot.engine.predict_next(prompt, top_k=5)
            dt_us = (time.perf_counter() - t0) * 1e6
            print(f"\033[1;36m[Prédiction Top-5 en {dt_us:.2f} µs]\033[0m pour '{prompt}' :")
            for rank, (tok, prob) in enumerate(candidates, 1):
                bar = "█" * int(prob * 30)
                print(f"  {rank}. \033[1m{tok:<15}\033[0m : {prob*100:5.1f}% | {bar}")
            continue

        # Commande /stats
        if cmd_lower == "/stats":
            s = bot.get_stats()
            print(f"\033[1;36m[Statistiques HiveSTA v2]\033[0m")
            print(f"  * Vocabulaire : {s['vocab_size']} tokens uniques")
            print(f"  * Contextes n-grams : {s['contexts_count']}")
            print(f"  * Transitions : {s['transitions_count']}")
            print(f"  * Documents en mémoire : {s['documents_count']}")
            continue

        if cmd_lower == "/help":
            print("Commandes : /learn <fait>, /predict <texte>, /stats, quit")
            continue

        # Réponse normale du chatbot
        t0 = time.perf_counter()
        response = bot.answer(user_input)
        dt_ms = (time.perf_counter() - t0) * 1000.0

        print(f"\033[1;34mHiveSTA \033[0;37m({dt_ms:.2f} ms)\033[1;34m :\033[0m {response}\n")


if __name__ == "__main__":
    run_interactive_cli()
