"""
Benchmark dataset: 100 research topics for evaluating the DeepArticle pipeline.

Each entry is a realistic query a researcher might type into the app. The set is
bilingual (≈50 English / ≈50 Turkish) and spans many domains so the benchmark
exercises query analysis, multi-source retrieval and summarization broadly.

Schema per item:
    id      : stable identifier
    query   : the research topic (as a user would type it)
    lang    : "en" | "tr"
    domain  : coarse research area (for slicing the report)
"""

from typing import List, Dict

BENCHMARK_QUESTIONS: List[Dict[str, str]] = [
    # ---- English (1-50) ----
    {"id": "en-01", "query": "retrieval augmented generation for question answering", "lang": "en", "domain": "nlp"},
    {"id": "en-02", "query": "parameter efficient fine-tuning of large language models", "lang": "en", "domain": "nlp"},
    {"id": "en-03", "query": "chain of thought prompting for reasoning", "lang": "en", "domain": "nlp"},
    {"id": "en-04", "query": "hallucination detection in large language models", "lang": "en", "domain": "nlp"},
    {"id": "en-05", "query": "mixture of experts transformer architectures", "lang": "en", "domain": "nlp"},
    {"id": "en-06", "query": "vision transformers for image classification", "lang": "en", "domain": "cv"},
    {"id": "en-07", "query": "diffusion models for image generation", "lang": "en", "domain": "cv"},
    {"id": "en-08", "query": "self-supervised contrastive learning for visual representations", "lang": "en", "domain": "cv"},
    {"id": "en-09", "query": "3D object detection for autonomous driving", "lang": "en", "domain": "cv"},
    {"id": "en-10", "query": "neural radiance fields for novel view synthesis", "lang": "en", "domain": "cv"},
    {"id": "en-11", "query": "graph neural networks for drug discovery", "lang": "en", "domain": "graph-ml"},
    {"id": "en-12", "query": "knowledge graph embedding for link prediction", "lang": "en", "domain": "graph-ml"},
    {"id": "en-13", "query": "deep reinforcement learning for robotic manipulation", "lang": "en", "domain": "rl"},
    {"id": "en-14", "query": "offline reinforcement learning algorithms", "lang": "en", "domain": "rl"},
    {"id": "en-15", "query": "multi-agent reinforcement learning coordination", "lang": "en", "domain": "rl"},
    {"id": "en-16", "query": "federated learning with differential privacy", "lang": "en", "domain": "ml-systems"},
    {"id": "en-17", "query": "model compression and knowledge distillation", "lang": "en", "domain": "ml-systems"},
    {"id": "en-18", "query": "quantization of deep neural networks for edge devices", "lang": "en", "domain": "ml-systems"},
    {"id": "en-19", "query": "neural architecture search methods", "lang": "en", "domain": "automl"},
    {"id": "en-20", "query": "out of distribution detection in deep learning", "lang": "en", "domain": "ml-theory"},
    {"id": "en-21", "query": "adversarial robustness of neural networks", "lang": "en", "domain": "ml-security"},
    {"id": "en-22", "query": "membership inference attacks on machine learning models", "lang": "en", "domain": "ml-security"},
    {"id": "en-23", "query": "explainable AI methods for model interpretability", "lang": "en", "domain": "xai"},
    {"id": "en-24", "query": "causal inference with machine learning", "lang": "en", "domain": "ml-theory"},
    {"id": "en-25", "query": "self-supervised learning for speech recognition", "lang": "en", "domain": "speech"},
    {"id": "en-26", "query": "neural machine translation for low resource languages", "lang": "en", "domain": "nlp"},
    {"id": "en-27", "query": "named entity recognition with transformers", "lang": "en", "domain": "nlp"},
    {"id": "en-28", "query": "sentiment analysis using deep learning", "lang": "en", "domain": "nlp"},
    {"id": "en-29", "query": "unit test generation using large language models", "lang": "en", "domain": "software-eng"},
    {"id": "en-30", "query": "automated program repair with deep learning", "lang": "en", "domain": "software-eng"},
    {"id": "en-31", "query": "code generation from natural language", "lang": "en", "domain": "software-eng"},
    {"id": "en-32", "query": "vulnerability detection in source code with machine learning", "lang": "en", "domain": "software-eng"},
    {"id": "en-33", "query": "fuzzing techniques for software security testing", "lang": "en", "domain": "security"},
    {"id": "en-34", "query": "intrusion detection systems using deep learning", "lang": "en", "domain": "security"},
    {"id": "en-35", "query": "malware classification with neural networks", "lang": "en", "domain": "security"},
    {"id": "en-36", "query": "blockchain consensus mechanisms scalability", "lang": "en", "domain": "distributed-systems"},
    {"id": "en-37", "query": "serverless computing performance optimization", "lang": "en", "domain": "systems"},
    {"id": "en-38", "query": "distributed training of large neural networks", "lang": "en", "domain": "ml-systems"},
    {"id": "en-39", "query": "vector databases for similarity search", "lang": "en", "domain": "databases"},
    {"id": "en-40", "query": "query optimization in relational databases", "lang": "en", "domain": "databases"},
    {"id": "en-41", "query": "recommender systems with graph neural networks", "lang": "en", "domain": "recsys"},
    {"id": "en-42", "query": "time series forecasting with deep learning", "lang": "en", "domain": "ml-apps"},
    {"id": "en-43", "query": "anomaly detection in network traffic", "lang": "en", "domain": "security"},
    {"id": "en-44", "query": "medical image segmentation with deep learning", "lang": "en", "domain": "medical-ai"},
    {"id": "en-45", "query": "protein structure prediction with deep learning", "lang": "en", "domain": "bioinformatics"},
    {"id": "en-46", "query": "quantum machine learning algorithms", "lang": "en", "domain": "quantum"},
    {"id": "en-47", "query": "continual learning to avoid catastrophic forgetting", "lang": "en", "domain": "ml-theory"},
    {"id": "en-48", "query": "few shot learning with meta learning", "lang": "en", "domain": "ml-theory"},
    {"id": "en-49", "query": "multimodal learning for vision and language", "lang": "en", "domain": "multimodal"},
    {"id": "en-50", "query": "energy efficient deep learning hardware accelerators", "lang": "en", "domain": "hardware"},

    # ---- Turkish (51-100) ----
    {"id": "tr-01", "query": "büyük dil modelleri ile soru cevaplama", "lang": "tr", "domain": "nlp"},
    {"id": "tr-02", "query": "derin öğrenme ile duygu analizi", "lang": "tr", "domain": "nlp"},
    {"id": "tr-03", "query": "Türkçe doğal dil işleme için dil modelleri", "lang": "tr", "domain": "nlp"},
    {"id": "tr-04", "query": "makine çevirisi için sinirsel ağlar", "lang": "tr", "domain": "nlp"},
    {"id": "tr-05", "query": "metin özetleme için derin öğrenme yöntemleri", "lang": "tr", "domain": "nlp"},
    {"id": "tr-06", "query": "evrişimli sinir ağları ile görüntü sınıflandırma", "lang": "tr", "domain": "cv"},
    {"id": "tr-07", "query": "nesne tespiti için derin öğrenme", "lang": "tr", "domain": "cv"},
    {"id": "tr-08", "query": "üretici çekişmeli ağlar ile görüntü üretimi", "lang": "tr", "domain": "cv"},
    {"id": "tr-09", "query": "yüz tanıma sistemlerinde derin öğrenme", "lang": "tr", "domain": "cv"},
    {"id": "tr-10", "query": "tıbbi görüntü bölütleme derin öğrenme", "lang": "tr", "domain": "medical-ai"},
    {"id": "tr-11", "query": "graf sinir ağları ile öneri sistemleri", "lang": "tr", "domain": "recsys"},
    {"id": "tr-12", "query": "pekiştirmeli öğrenme ile robot kontrolü", "lang": "tr", "domain": "rl"},
    {"id": "tr-13", "query": "federatif öğrenme ve gizlilik", "lang": "tr", "domain": "ml-systems"},
    {"id": "tr-14", "query": "derin sinir ağlarında model sıkıştırma", "lang": "tr", "domain": "ml-systems"},
    {"id": "tr-15", "query": "açıklanabilir yapay zeka yöntemleri", "lang": "tr", "domain": "xai"},
    {"id": "tr-16", "query": "yapay zeka ile siber güvenlik saldırı tespiti", "lang": "tr", "domain": "security"},
    {"id": "tr-17", "query": "makine öğrenmesi ile kötü amaçlı yazılım tespiti", "lang": "tr", "domain": "security"},
    {"id": "tr-18", "query": "ağ trafiğinde anomali tespiti", "lang": "tr", "domain": "security"},
    {"id": "tr-19", "query": "yazılım test üretimi için büyük dil modelleri", "lang": "tr", "domain": "software-eng"},
    {"id": "tr-20", "query": "kaynak kodda güvenlik açığı tespiti makine öğrenmesi", "lang": "tr", "domain": "software-eng"},
    {"id": "tr-21", "query": "otomatik program onarımı derin öğrenme", "lang": "tr", "domain": "software-eng"},
    {"id": "tr-22", "query": "zaman serisi tahmini derin öğrenme", "lang": "tr", "domain": "ml-apps"},
    {"id": "tr-23", "query": "makine öğrenmesi ile hastalık teşhisi", "lang": "tr", "domain": "medical-ai"},
    {"id": "tr-24", "query": "derin öğrenme ile kanser tespiti", "lang": "tr", "domain": "medical-ai"},
    {"id": "tr-25", "query": "elektronik sağlık kayıtları ile makine öğrenmesi", "lang": "tr", "domain": "medical-ai"},
    {"id": "tr-26", "query": "konuşma tanıma için derin öğrenme", "lang": "tr", "domain": "speech"},
    {"id": "tr-27", "query": "büyük dil modellerinde halüsinasyon", "lang": "tr", "domain": "nlp"},
    {"id": "tr-28", "query": "transfer öğrenme ile görüntü sınıflandırma", "lang": "tr", "domain": "cv"},
    {"id": "tr-29", "query": "çekişmeli örneklere karşı sağlamlık", "lang": "tr", "domain": "ml-security"},
    {"id": "tr-30", "query": "az örnekle öğrenme yöntemleri", "lang": "tr", "domain": "ml-theory"},
    {"id": "tr-31", "query": "graf gömme ile bağlantı tahmini", "lang": "tr", "domain": "graph-ml"},
    {"id": "tr-32", "query": "öznitelik seçimi makine öğrenmesi", "lang": "tr", "domain": "ml-theory"},
    {"id": "tr-33", "query": "topluluk öğrenmesi yöntemleri", "lang": "tr", "domain": "ml-theory"},
    {"id": "tr-34", "query": "derin öğrenme ile trafik akışı tahmini", "lang": "tr", "domain": "ml-apps"},
    {"id": "tr-35", "query": "akıllı şehirler için nesnelerin interneti", "lang": "tr", "domain": "iot"},
    {"id": "tr-36", "query": "kenar bilişimde yapay zeka uygulamaları", "lang": "tr", "domain": "systems"},
    {"id": "tr-37", "query": "blok zinciri ölçeklenebilirlik çözümleri", "lang": "tr", "domain": "distributed-systems"},
    {"id": "tr-38", "query": "bulut bilişimde kaynak yönetimi", "lang": "tr", "domain": "systems"},
    {"id": "tr-39", "query": "veri madenciliği ile müşteri segmentasyonu", "lang": "tr", "domain": "data-mining"},
    {"id": "tr-40", "query": "büyük veri analitiği yöntemleri", "lang": "tr", "domain": "data-mining"},
    {"id": "tr-41", "query": "doğal dil işleme ile sahte haber tespiti", "lang": "tr", "domain": "nlp"},
    {"id": "tr-42", "query": "derin öğrenme ile deprem tahmini", "lang": "tr", "domain": "ml-apps"},
    {"id": "tr-43", "query": "tarımda yapay zeka uygulamaları", "lang": "tr", "domain": "ml-apps"},
    {"id": "tr-44", "query": "enerji tüketimi tahmini makine öğrenmesi", "lang": "tr", "domain": "ml-apps"},
    {"id": "tr-45", "query": "finansal dolandırıcılık tespiti makine öğrenmesi", "lang": "tr", "domain": "ml-apps"},
    {"id": "tr-46", "query": "otonom araçlar için derin pekiştirmeli öğrenme", "lang": "tr", "domain": "rl"},
    {"id": "tr-47", "query": "protein yapısı tahmini derin öğrenme", "lang": "tr", "domain": "bioinformatics"},
    {"id": "tr-48", "query": "kuantum makine öğrenmesi algoritmaları", "lang": "tr", "domain": "quantum"},
    {"id": "tr-49", "query": "çok kipli öğrenme görüntü ve metin", "lang": "tr", "domain": "multimodal"},
    {"id": "tr-50", "query": "sürekli öğrenme ve felaket unutma", "lang": "tr", "domain": "ml-theory"},
]


def get_questions(lang: str = "all", limit: int = 0) -> List[Dict[str, str]]:
    """Return benchmark questions, optionally filtered by language and capped."""
    items = BENCHMARK_QUESTIONS
    if lang and lang != "all":
        items = [q for q in items if q["lang"] == lang]
    if limit and limit > 0:
        items = items[:limit]
    return items
