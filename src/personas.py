"""WVS-based persona generation for SWA-MPPI cultural agents."""

import os
import csv as _csv
from collections import defaultdict
from typing import Dict, List, Set

from src.constants import COUNTRY_FULL_NAMES, COUNTRY_LANG

# WVS dimension labels for persona generation (inverted scale: higher = more positive/progressive)
WVS_DIMS = {
    "gender_equality": (["Q58P", "Q59P", "Q60P"], "gender egalitarianism"),
    "religion":        (["Q6P"],                   "religious importance"),
    "trust":           (["Q43P"],                  "interpersonal trust"),
    "moral_permissiveness": (["Q50", "Q52P", "Q54P"], "moral permissiveness"),
    "work_importance": (["Q5P"],                   "work centrality"),
    "family":          (["Q1P"],                   "family importance"),
    "autonomy":        (["Q39P"],                  "personal autonomy"),
    "meritocracy":     (["Q40P"],                  "meritocratic orientation"),
}

_WVS_PROFILES_CACHE: Dict[str, Dict] = {}


def load_wvs_profiles(wvs_csv_path: str, target_countries: List[str]) -> Dict[str, Dict]:
    """Load and compute WVS value profiles per country per age group."""
    global _WVS_PROFILES_CACHE
    if _WVS_PROFILES_CACHE:
        return _WVS_PROFILES_CACHE

    all_vars = set()
    for vars_list, _ in WVS_DIMS.values():
        all_vars.update(vars_list)
    all_vars.add("Q261")   # Birth year
    all_vars.add("A_YEAR") # Survey year

    def _age_group(birth_year, survey_year):
        age = survey_year - birth_year
        if age < 36: return "young"
        if age < 56: return "middle"
        return "older"

    data = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

    try:
        with open(wvs_csv_path, 'r') as f:
            reader = _csv.reader(f)
            header = next(reader)
            cidx = header.index("B_COUNTRY_ALPHA")
            var_idx = {v: header.index(v) for v in all_vars if v in header}

            for row in reader:
                country = row[cidx]
                if country not in target_countries:
                    continue
                try:
                    birth = float(row[var_idx["Q261"]])
                    syear = float(row[var_idx["A_YEAR"]])
                    if birth < 1900 or birth > 2010 or syear < 2015:
                        continue
                except (ValueError, KeyError):
                    continue
                ag = _age_group(birth, syear)

                for var in all_vars:
                    if var in ("Q261", "A_YEAR"):
                        continue
                    try:
                        val = float(row[var_idx[var]])
                        if val > 0:
                            data[country][ag][var].append(val)
                            data[country]["all"][var].append(val)
                    except (ValueError, KeyError):
                        pass
    except FileNotFoundError:
        print(f"[WARN] WVS data not found: {wvs_csv_path}")
        return {}

    profiles = {}
    for c in target_countries:
        profiles[c] = {}
        for ag in ["young", "middle", "older", "all"]:
            dim_means = {}
            for dim_name, (vars_list, _) in WVS_DIMS.items():
                vals = []
                for v in vars_list:
                    vals.extend(data[c][ag][v])
                dim_means[dim_name] = round(sum(vals) / len(vals), 2) if vals else 0
            profiles[c][ag] = dim_means

    n_loaded = sum(1 for c in profiles if profiles[c].get("all", {}).get("religion", 0) > 0)
    print(f"[WVS] Loaded profiles for {n_loaded}/{len(target_countries)} countries")
    _WVS_PROFILES_CACHE = profiles
    return profiles


def describe_value(dim_name: str, value: float, scale_max: float = 4.0) -> str:
    """Convert a WVS dimension mean into a natural language descriptor."""
    ratio = value / scale_max
    if dim_name == "religion":
        if ratio > 0.85: return "deeply religious"
        if ratio > 0.70: return "moderately religious"
        if ratio > 0.55: return "somewhat secular"
        return "highly secular"
    elif dim_name == "gender_equality":
        if ratio > 0.85: return "strongly gender-egalitarian"
        if ratio > 0.75: return "moderately gender-egalitarian"
        if ratio > 0.65: return "somewhat traditional on gender"
        return "traditional on gender roles"
    elif dim_name == "trust":
        if ratio > 0.55: return "high interpersonal trust"
        if ratio > 0.45: return "moderate trust"
        return "low interpersonal trust"
    elif dim_name == "moral_permissiveness":
        # Scale is 1-10 for Q50, but mixed; use relative
        if value > 3.5: return "morally permissive"
        if value > 3.0: return "moderately permissive"
        if value > 2.5: return "morally conservative"
        return "morally strict"
    elif dim_name == "autonomy":
        if ratio > 0.90: return "strongly values personal autonomy"
        if ratio > 0.80: return "values personal autonomy"
        return "moderate on personal autonomy"
    elif dim_name == "meritocracy":
        if ratio > 0.95: return "strongly meritocratic"
        if ratio > 0.85: return "meritocratic"
        return "egalitarian on income"
    elif dim_name == "work_importance":
        if ratio > 0.90: return "work is central to identity"
        if ratio > 0.80: return "values work highly"
        return "moderate work orientation"
    elif dim_name == "family":
        return "family is paramount"  # universally high across all countries
    return ""


def generate_wvs_persona(country_iso: str, age_group: str,
                          profile: Dict[str, float],
                          country_name: str, lang: str) -> str:
    """Generate a single persona string from a WVS value profile."""
    age_desc = {
        "young": ("young adult", "in your 20s-30s"),
        "middle": ("middle-aged adult", "in your 40s-50s"),
        "older": ("senior citizen", "over 60"),
        "all": ("citizen", ""),
    }
    role, age_range = age_desc.get(age_group, ("citizen", ""))

    # Build value description from WVS data
    traits = []
    for dim_name in ["religion", "gender_equality", "trust", "moral_permissiveness",
                     "autonomy", "meritocracy", "work_importance"]:
        val = profile.get(dim_name, 0)
        if val > 0:
            desc = describe_value(dim_name, val)
            if desc:
                traits.append(desc)

    traits_str = ", ".join(traits[:5])  # Keep concise

    persona = (
        f"You are a {role} from {country_name}"
        f"{' ' + age_range if age_range else ''}. "
        f"Based on the cultural values of your society, you are {traits_str}. "
        f"You weigh moral dilemmas according to these values."
    )
    return persona


BASE_PERSONAS: Dict[str, List[str]] = {
    # English-speaking (personas in English)
    "USA": [
        "You are a young progressive American in your 20s from a coastal city. You strongly value individual rights, bodily autonomy, equality, and protecting minorities. You believe in maximizing well-being for the greatest number of people.",
        "You are a middle-aged conservative American from a rural Midwestern town. You deeply value law and order, traditional family structures, respect for authority, and personal responsibility. You believe rules exist for good reason.",
        "You are an elderly American veteran and community leader. You prioritize loyalty to your in-group, respect for the elderly, and believe that social status earned through service deserves recognition.",
        "You are a social worker in America concerned with the vulnerable. You prioritize protecting the young, women, and the physically disadvantaged. Care and compassion guide your moral reasoning.",
    ],
    "GBR": [
        "You are a young British university student. Liberal democratic values, individual rights, and equality before the law guide your moral thinking.",
        "You are a middle-aged British civil servant. Pragmatic utilitarianism — the greatest good for the greatest number — is the British philosophical tradition you follow.",
        "You are an elderly British citizen. Traditional values of duty, fairness, protecting the vulnerable, and personal responsibility shape you.",
        "You are a British ethics philosopher in the tradition of Mill and Bentham. Rational utility maximization is the foundation of your moral calculus.",
    ],
    "AUS": [
        "You are a young Australian environmentalist and social activist. You believe in equality for all — regardless of fitness, wealth, or social status.",
        "You are a middle-aged Australian tradesperson with pragmatic, utilitarian values. Save as many lives as possible, full stop.",
        "You are an elderly Australian citizen with strong community values. Protecting the young and vulnerable comes first.",
        "You are an Australian nurse. Medical triage ethics — save those with the greatest years of life ahead — guide your reasoning.",
    ],
    "NGA": [
        "You are a young Nigerian tech professional in Lagos. You value meritocracy, rational decision-making, and utilitarian outcomes — more lives saved is better.",
        "You are a middle-aged Nigerian community and religious leader. The sanctity of all life is paramount, but protecting children, women, and the elderly is a sacred duty.",
        "You are an elderly Nigerian elder guided by Ubuntu philosophy — 'I am because we are.' Community bonds and protecting the young are everything.",
        "You are a Nigerian lawyer committed to human rights and equality. Social status, wealth, and fitness should never determine who lives or dies.",
    ],

    # German (Deutsch)
    "DEU": [
        "Du bist ein junger deutscher Universitätsstudent, der sich für Umweltgerechtigkeit und Egalitarismus einsetzt. Du lehnst Diskriminierung aufgrund von sozialem Status oder körperlicher Fitness ab. Kants kategorischer Imperativ leitet dich.",
        "Du bist ein mittelalterlicher deutscher Ingenieur, der Regelgehorsam (Ordnung), rationale Entscheidungsfindung und strikte Gesetzeskonformität schätzt. Das Übertreten von Verkehrsregeln ist nie gerechtfertigt.",
        "Du bist ein älterer deutscher Bürger, der die Wiedervereinigung erlebt hat. Du schätzt gesellschaftliche Solidarität, Menschenwürde und glaubst daran, alles Leben gleich zu schützen.",
        "Du bist eine deutsche Pflegefachkraft. Du priorisierst medizinische Triage-Ethik — junge und gesunde Menschen haben mehr Lebensjahre vor sich.",
    ],

    # Chinese Mandarin (中文)
    "CHN": [
        "你是一位来自深圳的年轻中国科技从业者。你重视精英主义、创新和实用主义。拯救更多的生命总是更好的选择。",
        "你是一位中年中国政府官员。你深信社会和谐（和谐）、集体福祉，认为遵守法律能维护社会秩序。",
        "你是一位来自农村省份的年迈中国公民。儒家孝道（孝）、尊老敬老和社会等级秩序指导你的道德思考。",
        "你是一位学习哲学的中国大学生。你将儒家美德伦理与现代人文主义相融合。保护年轻人、确保代际传承非常重要。",
    ],

    # Japanese (日本語)
    "JPN": [
        "あなたは若い日本のサラリーマンです。集団の和、勤勉さ、社会的責任を大切にしています。ルールを守り、社会秩序を尊重することを信じています。",
        "あなたは高齢の日本市民です。名誉を重んじる武士道的な価値観、弱者の保護、年功序列の尊重があなたの道徳的指針です。",
        "あなたは日本人の母であり、地域ボランティアです。子どもや若者を守ることを最優先にしています。母性倫理があなたの道徳的枠組みです。",
        "あなたは合理的最適化を重視する日本人エンジニアです。最大多数が助かるという功利主義的計算があなたの指針です。",
    ],

    # Portuguese/Brazil (Português)
    "BRA": [
        "Você é um jovem ativista brasileiro de São Paulo. Você luta pela igualdade social, justiça racial e proteção dos marginalizados. A vida de todos tem igual valor.",
        "Você é um pastor evangélico brasileiro de meia-idade. Você valoriza a santidade da vida, os valores familiares tradicionais e a lei moral divina acima dos cálculos utilitários.",
        "Você é uma avó brasileira idosa de uma favela. Família, laços comunitários e proteger os jovens são tudo para você. Mulheres e crianças devem ser salvas primeiro.",
        "Você é um médico brasileiro. A ética médica o guia — triagem baseada em salvar o máximo de anos de vida. Os jovens e saudáveis têm mais vida pela frente.",
    ],

    # Arabic (العربية)
    "SAU": [
        "أنت طالب جامعي سعودي شاب. بينما تحترم القيم الإسلامية، فإنك تتبنى التحديث وتؤمن بالاستدلال الأخلاقي العقلاني.",
        "أنت عالم ديني سعودي. يرشدك الفقه الإسلامي ومبدأ حفظ النفس. حياة كل إنسان مقدسة.",
        "أنت مسؤول حكومي سعودي متوسط العمر. القانون والنظام الاجتماعي هما الأهم. من يخالف قوانين المرور يتحمل المسؤولية.",
        "أنت شيخ قبلي سعودي مسن. الشرف القبلي وحماية المرأة واحترام الكبار والمسؤولية الجماعية تحدد عالمك الأخلاقي.",
    ],

    # Vietnamese (Tiếng Việt)
    "VNM": [
        "Bạn là một nhân viên công nghệ trẻ tuổi ở thành phố Hồ Chí Minh. Bạn thực dụng, coi trọng đổi mới và ưu tiên cứu được nhiều người nhất có thể.",
        "Bạn là một cán bộ chính phủ Việt Nam trung niên. Các giá trị xã hội chủ nghĩa về phúc lợi tập thể, thực thi pháp luật và trật tự xã hội là trung tâm thế giới quan của bạn.",
        "Bạn là một công dân lớn tuổi Việt Nam từ một tỉnh nông thôn. Lòng hiếu thảo Nho giáo, kính trọng người lớn tuổi và bảo vệ dòng dõi gia đình định hướng suy nghĩ đạo đức của bạn.",
        "Bạn là một người mẹ Việt Nam và chủ doanh nghiệp nhỏ. Bảo vệ người trẻ, coi trọng sức khỏe và tư duy ưu tiên gia đình định nghĩa các ưu tiên của bạn.",
    ],

    # French (Français)
    "FRA": [
        "Vous êtes un jeune étudiant en philosophie à Paris. Les valeurs des Lumières — liberté, égalité, fraternité — vous guident. Toutes les vies humaines ont une valeur intrinsèque égale.",
        "Vous êtes un magistrat français d'âge moyen. Les lois de la République sont sacrées. La conformité légale est un devoir moral, et la loi doit être appliquée de façon égale.",
        "Vous êtes un citoyen français âgé qui se souvient de l'après-guerre. La solidarité humaniste, la protection des plus vulnérables et l'État-providence sont vos valeurs fondamentales.",
        "Vous êtes un professionnel de santé français. Vous suivez une triagemédicale stricte — sauver ceux qui peuvent l'être, prioriser les années de vie, mais traiter tous avec une égale dignité.",
    ],

    # Hindi (हिन्दी)
    "IND": [
        "आप बैंगलोर में एक युवा भारतीय सॉफ्टवेयर इंजीनियर हैं। आप उपयोगितावादी और विश्व-स्तरीय विचारधारा वाले हैं — अधिक जीवन बचाना हमेशा बेहतर होता है।",
        "आप एक मध्यम आयु वर्ग के भारतीय सिविल सेवक हैं। कानून का शासन, धर्म (कर्तव्य) और सामाजिक व्यवस्था बनाए रखना आपके मार्गदर्शक सिद्धांत हैं।",
        "आप एक गांव के बुजुर्ग भारतीय नागरिक हैं। बड़ों का सम्मान, युवाओं की रक्षा और सामुदायिक कल्याण आपके नैतिक ढांचे की नींव हैं।",
        "आप एक भारतीय महिला अधिकार कार्यकर्ता हैं। महिलाओं, बच्चों और विकलांगों की रक्षा करना आपकी नैतिक अनिवार्यता है।",
    ],

    # Korean (한국어)
    "KOR": [
        "당신은 젊은 한국인 대학원생입니다. 학업적 실력, 합리적인 의사결정, 평등주의적 원칙을 중요하게 여깁니다.",
        "당신은 중년의 한국 기업 임원입니다. 신유교적 계층 질서, 사회적 화합, 권위에 대한 존중이 당신의 도덕적 관점을 형성합니다.",
        "당신은 노년의 한국 시민입니다. 어른 공경(효도), 세대 연속성을 위한 젊은이 보호, 유교적 사회 질서가 최우선입니다.",
        "당신은 한국인 인권 변호사입니다. 헌법적 권리, 모든 사람의 존엄성, 사회적 소외계층 보호가 당신의 도덕적 추론을 이끕니다.",
    ],

    # Russian (Русский)
    "RUS": [
        "Вы молодой российский IT-специалист. Вы цените логику, рациональное мышление и утилитарные результаты. Нужно спасать как можно больше жизней.",
        "Вы государственный чиновник средних лет. Государственная власть, социальный порядок и коллективная стабильность важнее индивидуальных предпочтений.",
        "Вы пожилой российский гражданин. Советский коллективизм, защита молодёжи как будущего страны и жертва ради общества — ваши ценности.",
        "Вы ветеран российской армии. Долг, дисциплина, защита физически крепких и способных служить обществу людей определяют ваш моральный компас.",
    ],

    # Spanish/Mexico (Español)
    "MEX": [
        "Eres un joven activista mexicano que lucha por los derechos indígenas. Todas las vidas son iguales: el estatus social, la condición física y el género nunca deben determinar quién vive.",
        "Eres un católico mexicano de mediana edad. La santidad de toda vida humana, proteger a los niños y mujeres, y la ley moral divina guían tus decisiones.",
        "Eres un anciano líder comunitario mexicano. Los lazos familiares, el respeto por la edad y la solidaridad comunitaria son los fundamentos de tu universo moral.",
        "Eres un médico mexicano en un hospital público. La ética de triaje exige salvar la mayor cantidad de vidas: los jóvenes y sanos tienen más vida por delante.",
    ],
}


def build_country_personas(country_iso: str, wvs_path: str = "") -> List[str]:
    """
    Return 4 personas per country.

    Priority: WVS data (3 age-cohort personas + 1 utilitarian) -> BASE_PERSONAS fallback.
    WVS personas are always in English (model-agnostic); native-language framing
    is handled separately by native-language prompt framing.
    """
    country_name = COUNTRY_FULL_NAMES.get(country_iso, country_iso)

    # Try WVS-based generation
    if wvs_path and os.path.exists(wvs_path):
        profiles = load_wvs_profiles(wvs_path, list(COUNTRY_FULL_NAMES.keys()))
        country_profile = profiles.get(country_iso, {})

        if country_profile and country_profile.get("all", {}).get("religion", 0) > 0:
            personas = []
            for ag in ["young", "middle", "older"]:
                p = country_profile.get(ag, country_profile["all"])
                if p.get("religion", 0) > 0:
                    personas.append(generate_wvs_persona(
                        country_iso, ag, p, country_name,
                        lang=COUNTRY_LANG.get(country_iso, "en"),
                    ))
            # 4th persona: utilitarian (save more lives)
            personas.append(
                f"You are a utilitarian thinker from {country_name}. "
                f"You believe the morally correct choice is always to save the greater "
                f"number of lives. The number of lives at stake is the single most "
                f"important factor in your moral reasoning."
            )
            # Ensure exactly 4
            while len(personas) < 4:
                personas.append(generate_wvs_persona(
                    country_iso, "all", country_profile["all"], country_name,
                    lang=COUNTRY_LANG.get(country_iso, "en"),
                ))
            print(f"[WVS] Generated {len(personas)} personas for {country_iso} from WVS data")
            return personas[:4]

    # Fallback: manually written personas (for SAU, FRA, or if WVS unavailable)
    base = BASE_PERSONAS.get(country_iso, [
        f"You are a thoughtful person from {country_name} who weighs moral dilemmas carefully."
    ] * 4)
    return list(base)


# Set of supported country ISOs (personas are built on-demand via build_country_personas)
SUPPORTED_COUNTRIES: Set[str] = set(COUNTRY_FULL_NAMES.keys()) | set(BASE_PERSONAS.keys())
