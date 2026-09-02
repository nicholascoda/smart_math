import streamlit as st
import math
import random
import matplotlib.pyplot as plt

# Configuração da página
st.set_page_config(page_title="Rede Neural do Zero - MLP", layout="wide")

st.title("🧠 Demonstração Prática: Rede Neural Multicamadas (MLP) do Zero")
st.markdown("""
Esta aplicação treina uma **Rede Neural Artificial do zero**, sem o uso de frameworks de IA.
Toda a matemática (*Forward Pass*, Cálculo de Erro Quadrático e *Backpropagation*) é executada via código Python puro.
""")

st.sidebar.header("⚙️ Parâmetros da Rede")

# Seleção do Problema / Tabela Verdade
problema = st.sidebar.selectbox(
    "Escolha o problema lógico:",
    ["XOR (OU Exclusivo)", "XNOR", "AND", "OR", "Personalizado"]
)

if problema == "XOR (OU Exclusivo)":
    dataset = [(0, 0, 0), (0, 1, 1), (1, 0, 1), (1, 1, 0)]
elif problema == "XNOR":
    dataset = [(0, 0, 1), (0, 1, 0), (1, 0, 0), (1, 1, 1)]
elif problema == "AND":
    dataset = [(0, 0, 0), (0, 1, 0), (1, 0, 0), (1, 1, 1)]
elif problema == "OR":
    dataset = [(0, 0, 0), (0, 1, 1), (1, 0, 1), (1, 1, 1)]
else:
    st.sidebar.subheader("Tabela Verdade Personalizada")
    t00 = st.sidebar.number_input("Saída para (0,0)", 0, 1, 0)
    t01 = st.sidebar.number_input("Saída para (0,1)", 0, 1, 1)
    t10 = st.sidebar.number_input("Saída para (1,0)", 0, 1, 1)
    t11 = st.sidebar.number_input("Saída para (1,1)", 0, 1, 0)
    dataset = [(0, 0, t00), (0, 1, t01), (1, 0, t10), (1, 1, t11)]

# Hiperparâmetros
lr = st.sidebar.slider("Taxa de Aprendizado (Learning Rate):", 0.01, 2.0, 0.8, step=0.05)
epochs = st.sidebar.slider("Número de Épocas:", 1000, 30000, 10000, step=1000)
seed = st.sidebar.number_input("Semente Aleatória (Seed):", value=1, step=1)

# Função de ativação Sigmoide
def sigmoid(z):
    return 1 / (1 + math.exp(-z))

# Botão de Treinamento
if st.button("🚀 Treinar Rede Neural"):
    random.seed(seed)
    
    # Inicialização dos Pesos e Viases
    w11 = random.uniform(-1, 1)
    w12 = random.uniform(-1, 1)
    b1 = 0.0

    w21 = random.uniform(-1, 1)
    w22 = random.uniform(-1, 1)
    b2 = 0.0

    v1 = random.uniform(-1, 1)
    v2 = random.uniform(-1, 1)
    c = 0.0

    def forward(x1, x2):
        h1 = sigmoid(w11 * x1 + w12 * x2 + b1)
        h2 = sigmoid(w21 * x1 + w22 * x2 + b2)
        o = sigmoid(v1 * h1 + v2 * h2 + c)
        return h1, h2, o

    def calcular_mse():
        total = 0.0
        for x1, x2, t in dataset:
            _, _, o = forward(x1, x2)
            total += (t - o) ** 2
        return total / len(dataset)

    # Historico para gráficos
    historico_epocas = []
    historico_perda = []

    # Loop de Treinamento
    for epoch in range(epochs):
        for x1, x2, t in dataset:
            # Forward
            h1, h2, o = forward(x1, x2)
            
            # Backpropagation (Gradiente Descendente)
            erro = t - o
            d_o = erro * o * (1 - o)
            
            # Atualiza camada de saída
            v1 += lr * d_o * h1
            v2 += lr * d_o * h2
            c += lr * d_o
            
            # Atualiza camada oculta
            d_h1 = d_o * v1 * h1 * (1 - h1)
            w11 += lr * d_h1 * x1
            w12 += lr * d_h1 * x2
            b1 += lr * d_h1
            
            d_h2 = d_o * v2 * h2 * (1 - h2)
            w21 += lr * d_h2 * x1
            w22 += lr * d_h2 * x2
            b2 += lr * d_h2

        if epoch % (epochs // 50) == 0 or epoch == epochs - 1:
            historico_epocas.append(epoch)
            historico_perda.append(calcular_mse())

    # Salva o estado treinado na sessão do Streamlit
    st.session_state['treinado'] = True
    st.session_state['modelo'] = (w11, w12, b1, w21, w22, b2, v1, v2, c)
    st.session_state['historico'] = (historico_epocas, historico_perda)
    st.session_state['dataset'] = dataset

# Exibição dos Resultados após Treinamento
if st.session_state.get('treinado', False):
    w11, w12, b1, w21, w22, b2, v1, v2, c = st.session_state['modelo']
    historico_epocas, historico_perda = st.session_state['historico']
    dataset = st.session_state['dataset']

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📊 Curva de Aprendizado (Erro MSE vs Épocas)")
        fig, ax = plt.subplots()
        ax.plot(historico_epocas, historico_perda, color="#1f77b4", linewidth=2)
        ax.set_xlabel("Épocas")
        ax.set_ylabel("Erro Quadrático Médio (MSE)")
        ax.grid(True, linestyle="--", alpha=0.6)
        st.pyplot(fig)

    with col2:
        st.subheader("🎯 Tabela de Resultados do Treinamento")
        resultados = []
        for x1, x2, t in dataset:
            h1 = sigmoid(w11 * x1 + w12 * x2 + b1)
            h2 = sigmoid(w21 * x1 + w22 * x2 + b2)
            o = sigmoid(v1 * h1 + v2 * h2 + c)
            resultados.append({
                "Entrada (x1, x2)": f"({x1}, {x2})",
                "Esperado (Target)": t,
                "Saída Obtida": f"{o:.4f}",
                "Classe Prevista": 1 if o >= 0.5 else 0
            })
        st.table(resultados)

    st.markdown("---")
    st.subheader("🧪 Teste Interativo ao Vivo")
    st.write("Mova os sliders abaixo para simular entradas arbitrárias no modelo treinado:")
    
    col_t1, col_t2, col_t3 = st.columns(3)
    with col_t1:
        input_x1 = st.slider("Entrada x1:", 0.0, 1.0, 0.0, step=0.1)
    with col_t2:
        input_x2 = st.slider("Entrada x2:", 0.0, 1.0, 1.0, step=0.1)
    
    with col_t3:
        h1 = sigmoid(w11 * input_x1 + w12 * input_x2 + b1)
        h2 = sigmoid(w21 * input_x1 + w22 * input_x2 + b2)
        pred_o = sigmoid(v1 * h1 + v2 * h2 + c)
        
        st.metric(label="Saída Contínua do Neurônio", value=f"{pred_o:.4f}")
        st.metric(label="Decisão Final (Limiar 0.5)", value=1 if pred_o >= 0.5 else 0)
