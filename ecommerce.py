# Classificação de Sentimentos em Reviews de E-commerce - VERSÃO COMPLETA
# Dataset: Brazilian E-Commerce Public Dataset by Olist
# Modelo: Gradient Boosting + Análise Completa de Palavras + Nuvens de Palavras

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.pipeline import Pipeline
import re
import string
from collections import Counter
from wordcloud import WordCloud
import warnings
warnings.filterwarnings('ignore')

# Configuração de visualização
plt.style.use('default')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10

print("🛍️ CLASSIFICAÇÃO DE SENTIMENTOS COM GRADIENT BOOSTING + ANÁLISE DE PALAVRAS")
print("=" * 80)

# 1. CARREGAMENTO E EXPLORAÇÃO DOS DADOS
print("\n📊 1. CARREGANDO E EXPLORANDO OS DADOS")
print("-" * 50)

# Carregando o dataset (assumindo que está no diretório atual)
try:
    df = pd.read_csv('olist_order_reviews_dataset.csv')
    print(f"✅ Dataset carregado com sucesso!")
    print(f"📏 Dimensões: {df.shape[0]} linhas x {df.shape[1]} colunas")
except FileNotFoundError:
    print("❌ Arquivo não encontrado. Criando dados de exemplo...")
    # Dados de exemplo mais realistas para demonstração
    exemplos_positivos = [
        "Produto excelente, chegou rápido e bem embalado! Recomendo muito!",
        "Qualidade incrível, superou minhas expectativas. Voltarei a comprar!",
        "Entrega super rápida, produto perfeito, atendimento nota 10!",
        "Maravilhoso! Exatamente como descrito, chegou antes do prazo.",
        "Ótima qualidade, preço justo, recomendo para todos!",
        "Produto top, embalagem cuidadosa, vendedor prestativo.",
        "Chegou rapidinho, qualidade excelente, muito satisfeito!",
        "Perfeito! Produto original, bem embalado, entrega rápida.",
        "Adorei! Qualidade surpreendente, preço bom, recomendo!",
        "Excelente produto, atendimento ágil, entrega no prazo!"
    ]
    
    exemplos_negativos = [
        "Produto péssimo, chegou quebrado e atendimento horrível!",
        "Qualidade ruim, demorou muito para chegar, decepcionado.",
        "Produto veio defeituoso, embalagem danificada, não recomendo.",
        "Atrasou muito, produto diferente da descrição, péssimo!",
        "Qualidade inferior, caro para o que oferece, arrependido.",
        "Chegou com defeito, atendimento não resolve, frustrante.",
        "Produto ruim, embalagem amassada, não vale o preço.",
        "Demorou semanas para chegar, produto com problema.",
        "Péssima experiência, produto quebrado, não comprem!",
        "Decepcionante, qualidade baixa, atendimento inexistente."
    ]
    
    exemplos_neutros = [
        "Produto ok, entrega normal, nada demais.",
        "Atende ao esperado, preço razoável, entrega no prazo.",
        "Produto comum, embalagem simples, cumpre o básico.",
        "Nem bom nem ruim, dentro do esperado para o preço.",
        "Produto mediano, entrega demorou um pouco mas chegou."
    ]
    
    # Gerar dataset sintético mais realista
    reviews_data = []
    np.random.seed(42)
    
    # Reviews positivos (notas 4 e 5)
    for i in range(400):
        score = np.random.choice([4, 5], p=[0.4, 0.6])
        text = np.random.choice(exemplos_positivos)
        reviews_data.append({
            'review_id': f'review_{i}',
            'review_score': score,
            'review_comment_message': text + f" Compra {i}."
        })
    
    # Reviews negativos (notas 1, 2, 3)
    for i in range(400, 700):
        score = np.random.choice([1, 2, 3], p=[0.4, 0.3, 0.3])
        text = np.random.choice(exemplos_negativos)
        reviews_data.append({
            'review_id': f'review_{i}',
            'review_score': score,
            'review_comment_message': text + f" Pedido {i}."
        })
    
    # Reviews neutros (nota 3)
    for i in range(700, 800):
        text = np.random.choice(exemplos_neutros)
        reviews_data.append({
            'review_id': f'review_{i}',
            'review_score': 3,
            'review_comment_message': text + f" Ordem {i}."
        })
    
    df = pd.DataFrame(reviews_data)

# Informações básicas do dataset
print(f"\n📋 INFORMAÇÕES BÁSICAS:")
print(df.info())

print(f"\n📊 PRIMEIRAS 5 LINHAS:")
print(df.head())

# Verificar valores nulos
print(f"\n❓ VALORES NULOS:")
print(df.isnull().sum())

# 2. ANÁLISE EXPLORATÓRIA
print("\n\n📈 2. ANÁLISE EXPLORATÓRIA DOS DADOS")
print("-" * 50)

# Distribuição das notas
if 'review_score' in df.columns:
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # Gráfico de barras
    score_counts = df['review_score'].value_counts().sort_index()
    ax1.bar(score_counts.index, score_counts.values, color='skyblue', alpha=0.8)
    ax1.set_title('Distribuição das Notas dos Reviews', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Nota')
    ax1.set_ylabel('Quantidade')
    ax1.grid(True, alpha=0.3)
    
    # Pizza
    ax2.pie(score_counts.values, labels=score_counts.index, autopct='%1.1f%%', 
            colors=['#ff9999','#66b3ff','#99ff99','#ffcc99','#ff99cc'])
    ax2.set_title('Proporção das Notas', fontsize=12, fontweight='bold')
    
    # Histograma
    ax3.hist(df['review_score'], bins=5, color='lightgreen', alpha=0.7, edgecolor='black')
    ax3.set_title('Histograma das Notas', fontsize=12, fontweight='bold')
    ax3.set_xlabel('Nota')
    ax3.set_ylabel('Frequência')
    ax3.grid(True, alpha=0.3)
    
    # Box plot
    ax4.boxplot(df['review_score'], patch_artist=True,
                boxprops=dict(facecolor='lightcoral', alpha=0.7))
    ax4.set_title('Box Plot das Notas', fontsize=12, fontweight='bold')
    ax4.set_ylabel('Nota')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    print(f"📊 ESTATÍSTICAS DAS NOTAS:")
    print(f"Média: {df['review_score'].mean():.2f}")
    print(f"Mediana: {df['review_score'].median():.2f}")
    print(f"Desvio Padrão: {df['review_score'].std():.2f}")
    print(f"Distribuição:\n{df['review_score'].value_counts().sort_index()}")

# 3. CRIAÇÃO DA VARIÁVEL ALVO
print("\n\n🎯 3. CRIAÇÃO DA VARIÁVEL ALVO (CLASSIFICAÇÃO BINÁRIA)")
print("-" * 50)

def classify_sentiment(score):
    """Classifica sentimento baseado na nota"""
    if score >= 4:
        return 'positivo'
    else:
        return 'negativo'

df['sentiment'] = df['review_score'].apply(classify_sentiment)

# Distribuição da variável alvo
sentiment_dist = df['sentiment'].value_counts()
print(f"📊 DISTRIBUIÇÃO DO SENTIMENTO:")
print(sentiment_dist)
print(f"\nProporção:")
print(f"Positivo: {sentiment_dist['positivo']/len(df)*100:.1f}%")
print(f"Negativo: {sentiment_dist['negativo']/len(df)*100:.1f}%")

# Visualização melhorada
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Gráfico de barras
sentiment_dist.plot(kind='bar', ax=ax1, color=['salmon', 'lightgreen'], alpha=0.8)
ax1.set_title('Distribuição dos Sentimentos', fontsize=12, fontweight='bold')
ax1.set_xlabel('Sentimento')
ax1.set_ylabel('Quantidade')
ax1.tick_params(axis='x', rotation=0)
ax1.grid(True, alpha=0.3)

# Pizza
ax2.pie(sentiment_dist.values, labels=sentiment_dist.index, autopct='%1.1f%%',
        colors=['salmon', 'lightgreen'], startangle=90)
ax2.set_title('Proporção dos Sentimentos', fontsize=12, fontweight='bold')

plt.tight_layout()
plt.show()

# 4. PRÉ-PROCESSAMENTO DE TEXTO
print("\n\n🔧 4. PRÉ-PROCESSAMENTO DE TEXTO")
print("-" * 50)

def preprocess_text(text):
    """Função aprimorada para pré-processar o texto"""
    if pd.isna(text):
        return ""
    
    # Converter para string e minúsculas
    text = str(text).lower()
    
    # Remover URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    
    # Remover menções e hashtags
    text = re.sub(r'@\w+|#\w+', '', text)
    
    # Manter apenas letras, números e espaços
    text = re.sub(r'[^a-zA-ZÀ-ÿ0-9\s]', ' ', text)
    
    # Remover números isolados
    text = re.sub(r'\b\d+\b', '', text)
    
    # Remover espaços extras
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

# Combinar colunas de texto disponíveis
text_columns = []
if 'review_comment_title' in df.columns:
    text_columns.append('review_comment_title')
if 'review_comment_message' in df.columns:
    text_columns.append('review_comment_message')

if text_columns:
    df['combined_text'] = df[text_columns].fillna('').agg(' '.join, axis=1)
else:
    df['combined_text'] = df['review_comment_message'].fillna('')

# Aplicar pré-processamento
df['processed_text'] = df['combined_text'].apply(preprocess_text)

# Remover linhas com texto muito curto
df_clean = df[df['processed_text'].str.len() > 5].copy()

print(f"✅ Pré-processamento concluído!")
print(f"📏 Dados após limpeza: {len(df_clean)} linhas")
print(f"📝 Texto médio por review: {df_clean['processed_text'].str.len().mean():.1f} caracteres")

# Mostrar exemplos
print(f"\n📝 EXEMPLOS DE TEXTO PRÉ-PROCESSADO:")
for i, (original, processed) in enumerate(zip(df_clean['combined_text'].head(3), 
                                            df_clean['processed_text'].head(3))):
    print(f"\nExemplo {i+1}:")
    print(f"Original: {original}")
    print(f"Processado: {processed}")

# 5. ANÁLISE DE PALAVRAS E NUVENS DE PALAVRAS
print("\n\n☁️ 5. ANÁLISE DE PALAVRAS E CRIAÇÃO DAS NUVENS DE PALAVRAS")
print("-" * 50)

def create_word_analysis_and_clouds(df_clean):
    """Cria análise detalhada de palavras e nuvens de palavras"""
    
    # Separar textos por sentimento
    positive_texts = df_clean[df_clean['sentiment'] == 'positivo']['processed_text']
    negative_texts = df_clean[df_clean['sentiment'] == 'negativo']['processed_text']
    all_texts = df_clean['processed_text']
    
    # Juntar todos os textos
    positive_corpus = ' '.join(positive_texts).lower()
    negative_corpus = ' '.join(negative_texts).lower()
    all_corpus = ' '.join(all_texts).lower()
    
    # Palavras de parada em português
    portuguese_stopwords = {
        'de', 'da', 'do', 'das', 'dos', 'em', 'na', 'no', 'nas', 'nos', 'para', 'por', 'com', 'sem',
        'sob', 'sobre', 'ate', 'desde', 'um', 'uma', 'uns', 'umas', 'o', 'a', 'os', 'as', 'e', 'ou',
        'mas', 'que', 'se', 'como', 'quando', 'onde', 'porque', 'nao', 'muito', 'mais', 'menos',
        'bem', 'mal', 'ja', 'ainda', 'so', 'todo', 'toda', 'todos', 'todas', 'este', 'esta',
        'estes', 'estas', 'esse', 'essa', 'esses', 'essas', 'aquele', 'aquela', 'aqueles', 'aquelas',
        'meu', 'minha', 'meus', 'minhas', 'seu', 'sua', 'seus', 'suas', 'nosso', 'nossa', 'nossos',
        'nossas', 'vou', 'vai', 'foi', 'era', 'ser', 'ter', 'estar', 'foi', 'compra', 'pedido',
        'produto', 'item', 'ordem', 'review'
    }
    
    print("📊 ESTATÍSTICAS DOS CORPORA:")
    print(f"  • Palavras em reviews positivos: {len(positive_corpus.split()):,}")
    print(f"  • Palavras em reviews negativos: {len(negative_corpus.split()):,}")
    print(f"  • Total de palavras: {len(all_corpus.split()):,}")
    
    # Contar palavras
    positive_words = [word for word in positive_corpus.split() 
                     if len(word) > 2 and word not in portuguese_stopwords]
    negative_words = [word for word in negative_corpus.split() 
                     if len(word) > 2 and word not in portuguese_stopwords]
    all_words = [word for word in all_corpus.split() 
                if len(word) > 2 and word not in portuguese_stopwords]
    
    pos_counter = Counter(positive_words)
    neg_counter = Counter(negative_words)
    all_counter = Counter(all_words)
    
    print(f"\n🔤 PALAVRAS ÚNICAS APÓS FILTROS:")
    print(f"  • Palavras únicas positivas: {len(pos_counter):,}")
    print(f"  • Palavras únicas negativas: {len(neg_counter):,}")
    print(f"  • Total palavras únicas: {len(all_counter):,}")
    
    # Criar nuvens de palavras
    print(f"\n☁️ GERANDO NUVENS DE PALAVRAS...")
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 15))
    
    # 1. Nuvem de palavras GERAL
    try:
        wordcloud_all = WordCloud(
            width=800, height=400,
            background_color='white',
            max_words=100,
            colormap='viridis',
            font_path=None,
            relative_scaling=0.5,
            min_font_size=10
        ).generate_from_frequencies(dict(all_counter.most_common(100)))
        
        ax1.imshow(wordcloud_all, interpolation='bilinear')
        ax1.axis('off')
        ax1.set_title('🌟 NUVEM DE PALAVRAS - TODAS AS REVIEWS', 
                     fontsize=14, fontweight='bold', pad=20)
    except Exception as e:
        ax1.text(0.5, 0.5, f'Erro na nuvem geral:\n{str(e)}', 
                ha='center', va='center', transform=ax1.transAxes)
        ax1.set_title('Nuvem Geral - Erro')
    
    # 2. Nuvem de palavras POSITIVAS
    try:
        wordcloud_pos = WordCloud(
            width=800, height=400,
            background_color='white',
            max_words=80,
            colormap='Greens',
            font_path=None,
            relative_scaling=0.5,
            min_font_size=10
        ).generate_from_frequencies(dict(pos_counter.most_common(80)))
        
        ax2.imshow(wordcloud_pos, interpolation='bilinear')
        ax2.axis('off')
        ax2.set_title('💚 NUVEM DE PALAVRAS - REVIEWS POSITIVOS', 
                     fontsize=14, fontweight='bold', pad=20, color='darkgreen')
    except Exception as e:
        ax2.text(0.5, 0.5, f'Erro na nuvem positiva:\n{str(e)}', 
                ha='center', va='center', transform=ax2.transAxes)
        ax2.set_title('Nuvem Positiva - Erro')
    
    # 3. Nuvem de palavras NEGATIVAS
    try:
        wordcloud_neg = WordCloud(
            width=800, height=400,
            background_color='white',
            max_words=80,
            colormap='Reds',
            font_path=None,
            relative_scaling=0.5,
            min_font_size=10
        ).generate_from_frequencies(dict(neg_counter.most_common(80)))
        
        ax3.imshow(wordcloud_neg, interpolation='bilinear')
        ax3.axis('off')
        ax3.set_title('❤️ NUVEM DE PALAVRAS - REVIEWS NEGATIVOS', 
                     fontsize=14, fontweight='bold', pad=20, color='darkred')
    except Exception as e:
        ax3.text(0.5, 0.5, f'Erro na nuvem negativa:\n{str(e)}', 
                ha='center', va='center', transform=ax3.transAxes)
        ax3.set_title('Nuvem Negativa - Erro')
    
    # 4. Gráfico de barras - Top palavras
    top_words_all = dict(all_counter.most_common(15))
    ax4.barh(range(len(top_words_all)), list(top_words_all.values()), 
             color='skyblue', alpha=0.8)
    ax4.set_yticks(range(len(top_words_all)))
    ax4.set_yticklabels(list(top_words_all.keys()))
    ax4.set_xlabel('Frequência')
    ax4.set_title('📊 TOP 15 PALAVRAS MAIS FREQUENTES', 
                 fontsize=14, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return pos_counter, neg_counter, all_counter

# Executar análise de palavras e criar nuvens
pos_counter, neg_counter, all_counter = create_word_analysis_and_clouds(df_clean)

# 6. PREPARAÇÃO DOS DADOS PARA MODELAGEM
print("\n\n⚙️ 6. PREPARAÇÃO DOS DADOS PARA GRADIENT BOOSTING")
print("-" * 50)

# Separar features e target
X = df_clean['processed_text']
y = df_clean['sentiment']

# Divisão treino/teste estratificada
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"📊 DIVISÃO DOS DADOS:")
print(f"Treino: {len(X_train)} amostras")
print(f"Teste: {len(X_test)} amostras")
print(f"\nDistribuição no treino:")
print(y_train.value_counts())
print(f"Distribuição no teste:")
print(y_test.value_counts())

# 7. MODELAGEM COM GRADIENT BOOSTING E COMPARAÇÃO
print("\n\n🤖 7. TREINAMENTO COM GRADIENT BOOSTING E OUTROS MODELOS")
print("-" * 50)

# Configurar TF-IDF
tfidf = TfidfVectorizer(
    max_features=5000,
    ngram_range=(1, 2),
    min_df=2,
    max_df=0.95,
    stop_words=None  # Já removemos as stop words no pré-processamento
)

# Definir modelos para comparação
models = {
    'Gradient Boosting': GradientBoostingClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=6,
        random_state=42,
        verbose=0
    ),
    'Random Forest': RandomForestClassifier(
        n_estimators=100, 
        random_state=42, 
        n_jobs=-1
    ),
    'Logistic Regression': LogisticRegression(
        random_state=42, 
        max_iter=1000,
        C=1.0
    ),
    'Naive Bayes': MultinomialNB(alpha=1.0)
}

results = {}
print("🔄 Treinando e avaliando modelos...")

for name, model in models.items():
    print(f"\n📚 Treinando {name}...")
    
    # Criar pipeline
    pipeline = Pipeline([
        ('tfidf', tfidf),
        ('classifier', model)
    ])
    
    # Treinamento
    pipeline.fit(X_train, y_train)
    
    # Predições
    y_pred = pipeline.predict(X_test)
    y_pred_proba = pipeline.predict_proba(X_test)
    
    # Métricas
    accuracy = accuracy_score(y_test, y_pred)
    
    # Cross-validation
    cv_scores = cross_val_score(pipeline, X_train, y_train, cv=5, scoring='accuracy')
    
    results[name] = {
        'model': pipeline,
        'accuracy': accuracy,
        'cv_mean': cv_scores.mean(),
        'cv_std': cv_scores.std(),
        'predictions': y_pred,
        'probabilities': y_pred_proba
    }
    
    print(f"✅ Acurácia: {accuracy:.4f}")
    print(f"📊 CV Score: {cv_scores.mean():.4f} (±{cv_scores.std() * 2:.4f})")

# 8. OTIMIZAÇÃO DO GRADIENT BOOSTING
print("\n\n🎯 8. OTIMIZAÇÃO DO GRADIENT BOOSTING")
print("-" * 50)

print("🔍 Executando Grid Search para otimizar Gradient Boosting...")

# Parâmetros para otimização
param_grid = {
    'classifier__n_estimators': [50, 100, 150],
    'classifier__learning_rate': [0.05, 0.1, 0.15],
    'classifier__max_depth': [4, 6, 8],
}

# Grid Search
gb_pipeline = Pipeline([
    ('tfidf', tfidf),
    ('classifier', GradientBoostingClassifier(random_state=42))
])

grid_search = GridSearchCV(
    gb_pipeline, 
    param_grid, 
    cv=3,
    scoring='accuracy',
    n_jobs=-1,
    verbose=1
)

grid_search.fit(X_train, y_train)

# Melhor modelo
best_gb_model = grid_search.best_estimator_
best_gb_pred = best_gb_model.predict(X_test)
best_gb_accuracy = accuracy_score(y_test, best_gb_pred)

print(f"🏆 MELHORES PARÂMETROS: {grid_search.best_params_}")
print(f"🎯 MELHOR SCORE CV: {grid_search.best_score_:.4f}")
print(f"📊 ACURÁCIA NO TESTE: {best_gb_accuracy:.4f}")

# Atualizar resultados com o melhor GB
results['Gradient Boosting Otimizado'] = {
    'model': best_gb_model,
    'accuracy': best_gb_accuracy,
    'cv_mean': grid_search.best_score_,
    'cv_std': 0,  # Não temos o std do grid search
    'predictions': best_gb_pred,
    'probabilities': best_gb_model.predict_proba(X_test)
}

# 9. COMPARAÇÃO DE RESULTADOS
print("\n\n📊 9. COMPARAÇÃO DOS MODELOS")
print("-" * 50)

# Criar DataFrame com resultados
results_df = pd.DataFrame({
    'Modelo': list(results.keys()),
    'Acurácia': [results[model]['accuracy'] for model in results.keys()],
    'CV Mean': [results[model]['cv_mean'] for model in results.keys()],
    'CV Std': [results[model]['cv_std'] for model in results.keys()]
})

print("🏆 RANKING DOS MODELOS:")
results_df_sorted = results_df.sort_values('Acurácia', ascending=False)
print(results_df_sorted.to_string(index=False, float_format='%.4f'))

# Visualização dos resultados
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# Gráfico de barras - Acurácia
ax1.bar(range(len(results_df_sorted)), results_df_sorted['Acurácia'], 
        color=['gold' if i == 0 else 'silver' if i == 1 else 'lightcoral' 
               for i in range(len(results_df_sorted))], alpha=0.8)
ax1.set_xticks(range(len(results_df_sorted)))
ax1.set_xticklabels(results_df_sorted['Modelo'], rotation=45, ha='right')
ax1.set_ylabel('Acurácia')
ax1.set_title('🏆 Comparação de Acurácia dos Modelos', fontweight='bold')
ax1.grid(True, alpha=0.3)

# Adicionar valores nas barras
for i, v in enumerate(results_df_sorted['Acurácia']):
    ax1.text(i, v + 0.005, f'{v:.3f}', ha='center', va='bottom', fontweight='bold')

# Gráfico de barras - CV Score
ax2.bar(range(len(results_df_sorted)), results_df_sorted['CV Mean'], 
        yerr=results_df_sorted['CV Std'], capsize=5,
        color=['gold' if i == 0 else 'silver' if i == 1 else 'lightblue' 
               for i in range(len(results_df_sorted))], alpha=0.8)
ax2.set_xticks(range(len(results_df_sorted)))
ax2.set_xticklabels(results_df_sorted['Modelo'], rotation=45, ha='right')
ax2.set_ylabel('CV Score')
ax2.set_title('📊 Cross-Validation Score', fontweight='bold')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Melhor modelo
best_model_name = results_df_sorted.iloc[0]['Modelo']
best_model = results[best_model_name]['model']

print(f"\n🥇 MELHOR MODELO: {best_model_name}")
print(f"📈 Acurácia: {results[best_model_name]['accuracy']:.4f}")

# 10. ANÁLISE DETALHADA DO MELHOR MODELO
print(f"\n\n📋 10. ANÁLISE DETALHADA - {best_model_name}")
print("-" * 50)

y_pred_best = results[best_model_name]['predictions']
print("📊 RELATÓRIO DE CLASSIFICAÇÃO:")
print(classification_report(y_test, y_pred_best, target_names=['Negativo', 'Positivo']))

# Matriz de confusão melhorada
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# Matriz de confusão
cm = confusion_matrix(y_test, y_pred_best)
sns