Pedro Henrique Chaves Lopo
# 🛍️ Análise de Sentimentos em Reviews de E-commerce

## 🎯 Objetivo do Projeto

Este projeto utiliza técnicas de Processamento de Linguagem Natural (PLN) e Machine Learning para construir um sistema capaz de classificar automaticamente reviews de produtos de um e-commerce como **positivos** ou **negativos**. O objetivo final não é apenas prever o sentimento, mas também extrair insights acionáveis que possam guiar decisões de negócio, como identificar os principais motivos de satisfação e insatisfação dos clientes.

## 🚀 Tecnologias Utilizadas

* **Python 3.x**
* **Pandas & NumPy:** Para manipulação e análise de dados.
* **Matplotlib & Seaborn:** Para visualização de dados.
* **Scikit-learn:** Para pré-processamento, modelagem de Machine Learning e avaliação.
* **Jupyter Notebook:** Como ambiente de desenvolvimento e apresentação da análise.

## 📖 Decisões de Projeto (Fluxo Linear)

Cada passo do projeto envolveu decisões importantes para garantir a qualidade e a relevância do resultado final. A seguir, o fluxo de decisões em ordem cronológica:

### 1. Definição do Problema e da Métrica Alvo

A primeira e mais crucial decisão foi transformar um problema de classificação de 5 estrelas em uma **classificação binária**.

* **Decisão:** Mapear as notas de 1 a 5 para as classes "positivo" e "negativo".
* **Justificativa:** Para o negócio, a distinção mais importante é saber se a experiência do cliente foi boa ou ruim. Uma classificação binária é mais direta para gerar alertas e ações corretivas. A regra de negócio definida foi:
    * **Notas 4 e 5 → `positivo`**: Indicam satisfação.
    * **Notas 1, 2 e 3 → `negativo`**: Indicam algum nível de insatisfação que merece atenção.

### 2. Fonte de Dados

* **Decisão:** Utilizar o dataset público [Brazilian E-Commerce Public Dataset by Olist](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce), focando no arquivo `olist_order_reviews_dataset.csv`.
* **Justificativa:** É um dataset robusto, com um grande volume de reviews reais em português, o que o torna ideal para treinar um modelo de análise de sentimento para o mercado brasileiro.

### 3. Pré-processamento e Limpeza de Texto

* **Decisão:** Criar uma função de pré-processamento (`preprocess_text`) para padronizar todo o texto dos reviews. As etapas foram:
    1.  Combinar título e mensagem do review para ter o máximo de contexto.
    2.  Converter todo o texto para minúsculas.
    3.  Remover toda a pontuação.
    4.  Remover números.
    5.  Remover espaços em branco extras.
* **Justificativa:** Modelos de Machine Learning não entendem texto bruto. Esta limpeza garante que palavras como "Produto", "produto" e "produto!" sejam tratadas como a mesma coisa, reduzindo a complexidade e melhorando a performance do modelo.

### 4. Engenharia de Features (Vetorização de Texto)

* **Decisão:** Utilizar a técnica **TF-IDF (Term Frequency-Inverse Document Frequency)** para converter o texto pré-processado em vetores numéricos.
* **Justificativa:** O TF-IDF foi escolhido em vez de uma simples contagem de palavras (CountVectorizer) porque ele atribui um peso maior às palavras que são importantes para um review específico, mas não tão comuns em todos os reviews. Isso ajuda o modelo a focar nos termos mais distintivos.
* **Hiperparâmetros:**
    * `ngram_range=(1, 2)`: Para que o modelo capture não apenas palavras isoladas ('ruim'), mas também expressões de duas palavras ('não recomendo', 'muito bom').
    * `max_features=5000`: Para limitar o vocabulário às 5000 palavras mais relevantes, controlando a dimensionalidade e evitando overfitting.

### 5. Modelagem e Treinamento

* **Decisão:** Treinar e comparar três algoritmos de classificação diferentes: `Logistic Regression`, `Random Forest` e `Multinomial Naive Bayes`.
* **Justificativa:** Não há um "melhor modelo" para todos os problemas. Comparar diferentes algoritmos permite selecionar empiricamente aquele que melhor se adapta a este dataset específico.
* **Validação:** A validação cruzada (`cross_val_score` com 5 folds) foi utilizada para obter uma estimativa mais robusta e confiável da performance de cada modelo, evitando que o resultado dependa de uma única divisão sorteada de treino/teste.

### 6. Avaliação

* **Decisão:** Utilizar a **Acurácia** como métrica principal para comparar os modelos, mas também analisar o **Relatório de Classificação (Precision, Recall, F1-Score)** e a **Matriz de Confusão** para o melhor modelo.
* **Justificativa:** A acurácia oferece uma visão geral rápida do desempenho. No entanto, analisar Precision e Recall é fundamental para entender como o modelo se comporta em cada classe (positiva e negativa) e identificar se ele está enviesado para uma delas.

### 7. Análise de Insights e Explicabilidade (XAI)

* **Decisão:** Ir além da classificação e implementar funcionalidades para extrair insights de negócio.
* **Justificativa:** O valor real de um modelo de IA no contexto de negócios não está apenas na predição, mas na capacidade de explicar o porquê e guiar ações. Por isso, foram desenvolvidas funções para:
    1.  **Identificar Palavras-Chave:** Mostrar quais termos mais influenciam uma decisão positiva ou negativa.
    2.  **Categorizar por Temas:** Agrupar os feedbacks em categorias como "Entrega", "Qualidade", etc.
    3.  **Criar um Dashboard de Satisfação:** Calcular KPIs para cada tema, permitindo que a área de negócio identifique rapidamente os pontos fortes e fracos da operação.

## ⚙️ Como Executar o Projeto

1.  **Clone o repositório:**
    ```bash
    git clone [https://github.com/seu-usuario/seu-repositorio.git](https://github.com/seu-usuario/seu-repositorio.git)
    ```
2.  **Navegue até a pasta do projeto:**
    ```bash
    cd seu-repositorio
    ```
3.  **Instale as dependências:**


## 📊 Resultados e Conclusões

Após o treinamento e avaliação, o modelo **[Nome do Melhor Modelo, ex: Logistic Regression]** foi selecionado como o de melhor performance, atingindo uma acurácia de **[Valor da Acurácia, ex: 94.2%]** no conjunto de teste.

A análise de insights revelou que os principais drivers de reviews negativos estão relacionados a **[Ex: atrasos na entrega e qualidade do produto]**, enquanto a **[Ex: rapidez na entrega e o bom custo-benefício]** são os pontos mais elogiados pelos clientes.

