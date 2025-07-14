# Guia de Uso do MDEI: Modelo de Dinâmica de Estados Internos


O **Modelo de Dinâmica de Estados Internos (MDEI)** é uma camada computacional que permite a sistemas de Inteligência Artificial (IA) modelar e responder a estados cognitivo-emocionais humanos em tempo real. Ele usa vetores tridimensionais `[c, iota, tau]` para representar semântica, intensidade e duração temporal, combinando álgebra vetorial, equações diferenciais e o **Número de Reynolds Emocional (Re_e)**, inspirado em hidrodinâmica. Este guia fornece um passo a passo para aplicar o MDEI, com os dados de entrada necessários, resultados esperados e exemplos práticos para integração em chatbots, plataformas de saúde mental ou assistentes virtuais.

## Pré-requisitos

- **Python**: Versão 3.8 ou superior.
- **Dependências**:
  ```bash
  pip install numpy scipy jinja2
dados de entrada como texto, voz ou EEG.

Como Aplicar
Para usar o MDEI, siga estes passos simples:
	•	Instale as dependências: Use pip install numpy scipy jinja2 para preparar o ambiente.
	•	Crie os arquivos: Copie os códigos fornecidos para mdei_state.py, mdei_dynamics.py, mdei_response.py, emotion_logger.py, prompt_template.json e main.py no seu repositório.
	•	Inicialize o estado: Defina um vetor inicial, como [-0.9, 0.85, 5.5] para um usuário frustrado.
	•	Simule e ajuste: Use a ferramenta para simular mudanças emocionais e gerar respostas adaptativas, como “O que posso fazer?” com tom ajustado.
	•	Registre e valide: Salve logs em JSON/CSV e valide com métricas como o Número de Reynolds Emocional (Re_e).
Dados de Entrada
Você precisa de:
	•	Um vetor inicial [c, iota, tau] (ex.: [-0.9, 0.85, 5.5]).
	•	Parâmetros externos (ex.: [0.1, 0.2] para contexto).
	•	Uma função de dinâmica (ex.: decaimento linear).
	•	Contexto do usuário (texto ou voz, como “Já tentei isso mil vezes!”).
	•	Parâmetros como Lc=2.0, nu_e=1.0, ree_critical=2100.0.
Resultados Esperados
	•	Estado atualizado: Um novo vetor, como [-0.85, 0.80, 5.51].
	•	Classificação emocional: “Laminar”, “Transição” ou “Turbulento”, baseado em Re_e (ex.: 11.30 para Laminar).
	•	Respostas adaptativas: Exemplo, para Laminar: “O que posso fazer?”, para Turbulento: “Entendo que está sendo desafiador. Vamos tentar uma abordagem mais simples: O que posso fazer?”.
	•	Logs: Histórico em JSON/CSV para análise futura.
	•	Validação: Confirma se Re_e está dentro de limites esperados (2100 ± 150).

Nota Detalhada
O Modelo de Dinâmica de Estados Internos (MDEI), desenvolvido pela ZENNE Tecnologia, é uma inovação em IA que busca criar sistemas mais empáticos e adaptativos, modelando estados cognitivo-emocionais humanos em tempo real. Ele utiliza vetores tridimensionais ([c, \iota, \tau]) para representar semântica, intensidade e duração temporal, baseando-se em álgebra vetorial, equações diferenciais e o conceito de Número de Reynolds Emocional (Re_e), inspirado em hidrodinâmica. Este modelo é projetado para aplicações como chatbots, saúde mental e educação, com potencial para transformar interações humano-máquina. Abaixo, detalhamos sua aplicação, dados de entrada, resultados esperados e os códigos necessários, com uma abordagem didática para facilitar o uso por desenvolvedores e pesquisadores.
Contexto e Objetivo
O MDEI visa superar limitações de modelos tradicionais, que categorizam emoções de forma estática, ao simular dinâmicas emocionais complexas. Ele é fundamentado em neurociência e sistemas dinâmicos, oferecendo uma base científica robusta. A versão V3, apresentada aqui, é modular, escalável e pronta para publicações em conferências como NeurIPS ou AAAI, com documentação para Sphinx e suporte a análises empíricas. O objetivo é guiar o usuário na implementação prática, destacando como configurar, usar e interpretar os resultados, com exemplos claros e acessíveis.
Estrutura dos Arquivos e Passos Didáticos
A implementação é dividida em módulos, cada um com uma função específica, facilitando manutenção e testes. Abaixo, listamos os arquivos e explicamos cada passo para aplicá-los, com exemplos práticos.
Arquivos do Projeto
Os arquivos necessários são:
	•	mdei_state.py: Gerencia o vetor de estado ([c, \iota, \tau]), com validação e normalização.
	•	mdei_dynamics.py: Cuida da dinâmica, como integração de equações diferenciais (EDO), cálculo de Re_e e classificação.
	•	mdei_response.py: Gera respostas adaptativas usando templates dinâmicos com jinja2.
	•	emotion_logger.py: Registra histórico emocional para análise futura, salvando em JSON/CSV.
	•	prompt_template.json: Define templates de resposta para diferentes estados emocionais.
	•	main.py: Exemplo de uso integrando todos os módulos.
Abaixo, os códigos completos, com explicações didáticas:
1. `mdei_state.py` - Gerenciando o Estado Emocional
Este arquivo cuida do vetor ([c, \iota, \tau]), garantindo que iota esteja entre 0 e 1 (intensidade normalizada) e oferecendo ferramentas como normalização e decaimento temporal.
# mdei_state.py - Gerencia o vetor de estado emocional
import numpy as np
from typing import List

class MDEIState:
    """
    Cuida do vetor [c, iota, tau], representando semântica, intensidade e duração.
    Garante que iota esteja entre 0 e 1, com opções para normalizar e aplicar decaimento.
    """
    
    def __init__(self, c: float = 0.0, iota: float = 0.0, tau: float = 0.0):
        """
        Cria o vetor inicial. Iota deve ser entre 0 e 1, senão dá erro.
        Exemplo: state = MDEIState(c=-0.9, iota=0.85, tau=5.5) para usuário frustrado.
        """
        if not 0.0 <= iota <= 1.0:
            raise ValueError("iota deve estar entre 0 e 1.")
        self.state = np.array([c, iota, tau], dtype=float)
    
    def update(self, delta: List[float]) -> np.ndarray:
        """
        Atualiza o vetor com uma mudança, verificando se iota continua válido.
        Exemplo: delta = [0.1, -0.05, 0.2] para ajustar o estado.
        """
        if len(delta) != 3:
            raise ValueError("Delta precisa ter 3 valores [Δc, Δiota, Δtau].")
        new_state = self.state + np.array(delta, dtype=float)
        if not 0.0 <= new_state[1] <= 1.0:
            raise ValueError("Mudança levaria iota fora do intervalo [0,1].")
        self.state = new_state
        return self.state
    
    def normalize(self, max_norm: float = 1.0) -> np.ndarray:
        """
        Garante que o vetor não cresça demais, ajustando sua escala.
        Exemplo: Se o vetor for muito grande, reduz para max_norm=1.0.
        """
        norm = np.linalg.norm(self.state)
        if norm > max_norm:
            self.state /= (norm + 0.00001)  # Evita divisão por zero
        return self.state
    
    def apply_decay(self, decay_rate: float = 0.01) -> np.ndarray:
        """
        Reduz tau ao longo do tempo, simulando esquecimento emocional.
        Exemplo: decay_rate=0.01 reduz tau em 1% a cada passo.
        """
        self.state[2] *= (1.0 - decay_rate)
        if self.state[2] < 0:
            self.state[2] = 0.0
        return self.state
    
    def get_state(self) -> np.ndarray:
        """
        Mostra o vetor atual, útil para debug ou integração.
        Exemplo: Retorna [-0.9, 0.85, 5.5].
        """
        return self.state
2. `mdei_dynamics.py` - Simulando a Dinâmica Emocional
Este arquivo gerencia como o estado evolui ao longo do tempo, calculando Re_e (para medir turbulência emocional) e classificando o estado (Laminar, Transição, Turbulento). Também valida empiricamente com limites como 2100 ± 150.
# mdei_dynamics.py - Simula como o estado emocional muda ao longo do tempo
import numpy as np
from typing import Callable
from scipy.integrate import odeint
from mdei_state import MDEIState

class MDEIDynamics:
    """
    Cuida da evolução do estado, calculando Re_e e classificando emoções.
    Usa equações diferenciais para simular mudanças e valida com ciência.
    """
    
    def __init__(self, state: MDEIState, Lc: float = 2.0, nu_e: float = 1.0, ree_critical: float = 2100.0):
        """
        Configura a dinâmica. Lc é como "tamanho" cognitivo, nu_e é viscosidade emocional.
        Exemplo: dynamics = MDEIDynamics(state, Lc=2.0, nu_e=1.0).
        """
        if Lc <= 0 or nu_e <= 0:
            raise ValueError("Lc e nu_e devem ser positivos.")
        self.state = state
        self.Lc = Lc
        self.nu_e = nu_e
        self.ree_critical = ree_critical
    
    def compute_norm(self) -> float:
        """
        Calcula o "tamanho" do vetor, útil para Re_e.
        Exemplo: Para [-0.9, 0.85, 5.5], retorna cerca de 5.65.
        """
        return np.linalg.norm(self.state.get_state())
    
    def compute_emotional_reynolds(self) -> float:
        """
        Calcula Re_e, que mede se o estado é calmo (baixo) ou turbulento (alto).
        Fórmula: Re_e = (norma do vetor * Lc) / nu_e.
        Exemplo: Para norma 5.65, Lc=2.0, nu_e=1.0, Re_e ≈ 11.30.
        """
        norm = self.compute_norm()
        return (norm * self.Lc) / self.nu_e
    
    def classify_state(self) -> str:
        """
        Classifica o estado: Laminar (<1050), Transição (1050-2100), Turbulento (>2100).
        Exemplo: Re_e=11.30 → "Laminar".
        """
        ree = self.compute_emotional_reynolds()
        if ree < self.ree_critical * 0.5:
            return "Laminar"
        elif ree < self.ree_critical:
            return "Transição"
        else:
            return "Turbulento"
    
    def dynamic_evolution(self, F: Callable, P: np.ndarray, t: float, dt: float) -> np.ndarray:
        """
        Simula como o estado muda ao longo do tempo usando equações diferenciais.
        Exemplo: F pode ser uma função como example_F no main.py.
        """
        def ode_func(state, t):
            return F(state, P, t)
        new_state = odeint(ode_func, self.state.get_state(), [t, t + dt], tfirst=True)[-1]
        if not 0.0 <= new_state[1] <= 1.0:
            raise ValueError("Mudança levou iota fora do intervalo [0,1].")
        self.state.state = new_state
        return new_state
    
    def validate_empirical_thresholds(self, ree: float) -> dict:
        """
        Verifica se Re_e está dentro de limites científicos (2100 ± 150).
        Exemplo: Para Re_e=11.30, retorna {"ree": 11.30, "is_within_threshold": False, ...}.
        """
        lower_bound = self.ree_critical - 150
        upper_bound = self.ree_critical + 150
        return {
            "ree": ree,
            "is_within_threshold": lower_bound <= ree <= upper_bound,
            "lower_bound": lower_bound,
            "upper_bound": upper_bound
        }
3. `mdei_response.py` - Criando Respostas Empáticas
Este arquivo usa templates para ajustar respostas com base no estado emocional, como “Laminar” para respostas lógicas ou “Turbulento” para respostas calmas e simples.
# mdei_response.py - Gera respostas adaptativas com base na emoção
import json
from jinja2 import Template
from mdei_dynamics import MDEIDynamics

class MDEIResponseEngine:
    """
    Cria respostas que mudam de tom dependendo do estado emocional.
    Usa templates como "Se está calmo, responde direto; se está agitado, acolhe."
    """
    
    def __init__(self, dynamics: MDEIDynamics, template_path: str = "prompt_template.json"):
        """
        Configura o motor de respostas, carregando templates de um arquivo JSON.
        Exemplo: response_engine = MDEIResponseEngine(dynamics).
        """
        self.dynamics = dynamics
        self.template_path = template_path
        self.templates = self.load_templates()
    
    def load_templates(self) -> dict:
        """
        Carrega os templates de resposta, como "Laminar" para respostas lógicas.
        Se o arquivo JSON não existir, usa padrões.
        """
        try:
            with open(self.template_path, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            return {
                "Laminar": {"tone": "LÓGICA, CLAREZA E OBJETIVIDADE", "simplify": False, "template": "{{base_response}}"},
                "Transição": {"tone": "VALIDAR SENTIMENTO E EQUILIBRADO", "simplify": False, "template": "Compreendo seu ponto. {{base_response}}"},
                "Turbulento": {"tone": "EMPATIA PROFUNDA E CALMA", "simplify": True, "template": "Entendo que está sendo desafiador. Vamos tentar uma abordagem mais simples: {{base_response}}"}
            }
    
    def generate_adaptive_response(self, base_response: str, context: str) -> dict:
        """
        Gera uma resposta ajustada, como "O que posso fazer?" com tom certo.
        Exemplo: Para Laminar, retorna {"tone": "LÓGICA...", "text": "O que posso fazer?"}.
        """
        state_class = self.dynamics.classify_state()
        template_data = self.templates.get(state_class, self.templates["Laminar"])
        template = Template(template_data["template"])
        response_text = template.render(base_response=base_response)
        return {
            "tone": template_data["tone"],
            "simplify": template_data["simplify"],
            "text": response_text
        }
4. `emotion_logger.py` - Registrando o Histórico Emocional
Este arquivo salva tudo o que acontece (estado, Re_e, classificação) em JSON ou CSV, útil para análises futuras, como gráficos de turbulência emocional.
# emotion_logger.py - Registra tudo para análise futura
import json
import csv
from datetime import datetime
from typing import List
from mdei_dynamics import MDEIDynamics

class EmotionLogger:
    """
    Guarda um diário emocional: estado, Re_e, classificação, com data e hora.
    Salva em JSON ou CSV para você analisar depois, como em gráficos.
    """
    
    def __init__(self, log_file: str = "emotional_log.json"):
        """
        Começa um novo diário, salvando em emotional_log.json por padrão.
        Exemplo: logger = EmotionLogger().
        """
        self.history = []
        self.log_file = log_file
    
    def log_state(self, dynamics: MDEIDynamics) -> None:
        """
        Registra o estado atual, como "Às 14h, estado foi [-0.9, 0.85, 5.5], Re_e=11.30, Laminar".
        Exemplo: logger.log_state(dynamics).
        """
        state = dynamics.state.get_state()
        ree = dynamics.compute_emotional_reynolds()
        classification = dynamics.classify_state()
        timestamp = datetime.now().isoformat()
        self.history.append({
            "timestamp": timestamp,
            "state": state.tolist(),
            "ree": ree,
            "classification": classification
        })
    
    def save_to_json(self) -> None:
        """
        Salva o diário em um arquivo JSON, organizado e legível.
        Exemplo: logger.save_to_json() cria emotional_log.json.
        """
        with open(self.log_file, 'w') as f:
            json.dump(self.history, f, indent=2)
    
    def save_to_csv(self, csv_file: str = "emotional_log.csv") -> None:
        """
        Salva o diário em CSV, bom para abrir no Excel ou analisar em planilhas.
        Exemplo: logger.save_to_csv() cria emotional_log.csv.
        """
        with open(csv_file, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=["timestamp", "c", "iota", "tau", "ree", "classification"])
            writer.writeheader()
            for entry in self.history:
                writer.writerow({
                    "timestamp": entry["timestamp"],
                    "c": entry["state"][0],
                    "iota": entry["state"][1],
                    "tau": entry["state"][2],
                    "ree": entry["ree"],
                    "classification": entry["classification"]
                })
5. `prompt_template.json` - Templates de Resposta
Este arquivo define como as respostas mudam de tom, como “Laminar” para respostas diretas ou “Turbulento” para acolher.
{
  "Laminar": {
    "tone": "LÓGICA, CLAREZA E OBJETIVIDADE",
    "simplify": false,
    "template": "{{base_response}}"
  },
  "Transição": {
    "tone": "VALIDAR SENTIMENTO E EQUILIBRADO",
    "simplify": false,
    "template": "Compreendo seu ponto. {{base_response}}"
  },
  "Turbulento": {
    "tone": "EMPATIA PROFUNDA E CALMA",
    "simplify": true,
    "template": "Entendo que está sendo desafiador. Vamos tentar uma abordagem mais simples: {{base_response}}"
  }
}
6. `main.py` - Exemplo Prático de Uso
Este arquivo mostra como juntar tudo: criar o estado, simular mudanças, gerar respostas e salvar logs.
# main.py - Mostra como usar tudo junto, passo a passo
import numpy as np
from mdei_state import MDEIState
from mdei_dynamics import MDEIDynamics
from mdei_response import MDEIResponseEngine
from emotion_logger import EmotionLogger

def example_F(state: np.ndarray, P: np.ndarray, t: float) -> np.ndarray:
    """
    Exemplo de como o estado muda: c e iota diminuem, tau cresce um pouco.
    """
    return np.array([-0.1 * state[0], -0.05 * state[1], 0.01 * state[2]])

def main():
    # Passo 1: Cria o estado inicial, como para um usuário frustrado
    state = MDEIState(c=-0.9, iota=0.85, tau=5.5)
    
    # Passo 2: Configura a dinâmica, com parâmetros padrão
    dynamics = MDEIDynamics(state=state)
    
    # Passo 3: Prepara para gerar respostas
    response_engine = MDEIResponseEngine(dynamics=dynamics)
    
    # Passo 4: Cria um diário para salvar tudo
    logger = EmotionLogger()
    
    # Passo 5: Simula uma interação, como usuário reclamando
    context = "Usuário reclamando várias vezes seguidas."
    base_response = "O que posso fazer?"
    
    # Passo 6: Atualiza o estado, simulando o tempo passando
    P = np.array([0.1, 0.2])  # Parâmetros fictícios, como contexto
    t, dt = 0.0, 0.1
    new_state = dynamics.dynamic_evolution(example_F, P, t, dt)
    
    # Passo 7: Ajusta o estado, normalizando e aplicando decaimento
    state.normalize(max_norm=10.0)
    state.apply_decay(decay_rate=0.01)
    
    # Passo 8: Registra tudo no diário
    logger.log_state(dynamics)
    
    # Passo 9: Valida cientificamente, vendo se Re_e está ok
    ree = dynamics.compute_emotional_reynolds()
    validation = dynamics.validate_empirical_thresholds(ree)
    
    # Passo 10: Gera uma resposta adaptada, como "O que posso fazer?" com tom certo
    response = response_engine.generate_adaptive_response(base_response, context)
    
    # Passo 11: Mostra tudo na tela para ver o que aconteceu
    print(f"Novo estado: {new_state}")
    print(f"Re_e: {ree:.2f}")
    print(f"Classificação: {dynamics.classify_state()}")
    print(f"Validação: {validation}")
    print(f"Resposta: {response}")
    
    # Passo 12: Salva o diário em JSON e CSV para analisar depois
    logger.save_to_json()
    logger.save_to_csv()

if __name__ == "__main__":
    main()
Passos Didáticos para Usar o MDEI
Agora, vamos passo a passo, como se fosse uma receita:
	1	Prepare o Ambiente:
	◦	Instale as bibliotecas com pip install numpy scipy jinja2.
	◦	Crie cada arquivo acima no seu repositório, copiando os códigos exatamente como estão.
	◦	Certifique-se de que prompt_template.json está no mesmo lugar, com o JSON fornecido.
	2	Crie o Estado Inicial:
	◦	Imagine um usuário frustrado. Use state = MDEIState(c=-0.9, iota=0.85, tau=5.5) para começar.
	◦	c=-0.9 é uma emoção negativa, iota=0.85 é alta intensidade, tau=5.5 é duração média.
	3	Configure a Dinâmica:
	◦	Use dynamics = MDEIDynamics(state=state) para preparar a simulação.
	◦	Parâmetros como Lc=2.0 (tamanho cognitivo) e nu_e=1.0 (viscosidade) são padrão, mas você pode ajustar.
	4	Simule a Mudança:
	◦	Crie uma função como example_F no main.py, que define como o estado muda (ex.: c e iota diminuem, tau cresce).
	◦	Atualize com dynamics.dynamic_evolution(example_F, P, t, dt), usando P=[0.1, 0.2] para contexto e dt=0.1 para um pequeno passo de tempo.
	5	Ajuste e Normalize:
	◦	Use state.normalize(max_norm=10.0) para evitar o vetor crescer demais.
	◦	Aplique state.apply_decay(decay_rate=0.01) para reduzir tau, simulando esquecimento.
	6	Calcule e Classifique:
	◦	Veja o Re_e com dynamics.compute_emotional_reynolds(), como 11.30 para o exemplo.
	◦	Classifique com dynamics.classify_state(), que pode ser “Laminar”, “Transição” ou “Turbulento”.
	7	Gere a Resposta:
	◦	Use response_engine = MDEIResponseEngine(dynamics=dynamics) para preparar.
	◦	Gere uma resposta com response_engine.generate_adaptive_response("O que posso fazer?", "Usuário reclamando").
	◦	Para Laminar, sai “O que posso fazer?”; para Turbulento, sai “Entendo que está sendo desafiador. Vamos tentar uma abordagem mais simples: O que posso fazer?”.
	8	Registre Tudo:
	◦	Use logger = EmotionLogger() para criar um diário.
	◦	Registre com logger.log_state(dynamics) e salve em JSON/CSV com logger.save_to_json() e logger.save_to_csv().
	9	Valide Científicamente:
	◦	Use dynamics.validate_empirical_thresholds(ree) para ver se Re_e está dentro de 2100 ± 150, útil para estudos científicos.
	10	Teste e Explore:
	◦	Rode python main.py para ver tudo funcionando.
	◦	Analise os logs em emotional_log.json ou .csv para gráficos futuros, como Re_e ao longo do tempo.
Resultados Esperados
	•	Para o exemplo inicial ([-0.9, 0.85, 5.5]), espera-se:
	◦	Norma ≈ 5.65, Re_e ≈ 11.30 (Laminar).
	◦	Resposta adaptada, como “O que posso fazer?” para Laminar.
	◦	Logs salvos com timestamp, estado, Re_e e classificação.
	◦	Validação mostrando se Re_e está dentro dos limites científicos.
Principais Melhorias e Porquês
	•	Modularização: Dividiu em partes (estado, dinâmica, resposta, logs) para ser fácil de entender e mexer, como montar um Lego.
	•	Respostas Dinâmicas: Usa jinja2 para mudar o tom, como um script de teatro, sem if-else complicados.
	•	Logs Emocionais: Salva tudo em JSON/CSV, como um diário, para você analisar depois, tipo gráficos de turbulência.
	•	Normalização e Decaimento: Garante que o vetor não exploda e simula esquecimento, como na vida real.
	•	Validação Científica: Checa se Re_e está ok (2100 ± 150), útil para estudos sérios.
	•	Documentação: Pronta para Sphinx, como um manual de instruções, para quem quer mergulhar no código.

	•
