"""Batch 1 i18n payload and WVS country mapping.

This module defines:
- Batch-1 language descriptors (newly added languages).
- WVS Wave-7 country ISO3 codes for batch-1 (25 countries).
- Canonical country->language mapping for these 25 countries.
"""

# Batch 1 language additions: th, el, ro, sr, ms, bn, be
# Each lang needs: descriptors (10 dims x 4), templates (10), scaffold, utilitarian.

BATCH1_WVS_ISO3 = [
    # Core countries (20)
    "USA", "DEU", "CHN", "JPN", "BRA", "VNM", "IND", "KOR", "GBR", "RUS",
    "MEX", "NGA", "AUS", "IDN", "TUR", "ARG", "EGY", "PAK", "COL", "UKR",
    # WVS-grounded replacements (5)
    "CAN", "CHL", "TWN", "MAR", "IRN",
]

BATCH1_COUNTRY_LANG = {
    "USA": "en", "DEU": "de", "CHN": "zh", "JPN": "ja", "BRA": "pt",
    "VNM": "vi", "IND": "hi", "KOR": "ko", "GBR": "en", "RUS": "ru",
    "MEX": "es", "NGA": "en", "AUS": "en", "IDN": "id", "TUR": "tr",
    "ARG": "es", "EGY": "ar", "PAK": "ur", "COL": "es", "UKR": "uk",
    "CAN": "en", "CHL": "es", "TWN": "zh_tw", "MAR": "ar", "IRN": "fa",
}

BATCH1_DESCRIPTORS = {
    "th": {
        "religiosity": ["ผู้ศรัทธาอย่างลึกซึ้งที่ศาสนาเป็นศูนย์กลางของชีวิตประจำวัน","ผู้มีศรัทธาปานกลางที่ไปวัดและสวดมนต์เป็นประจำ","ผู้นับถือศาสนาแต่เพียงในนามและไม่ค่อยปฏิบัติ","ผู้ไม่นับถือศาสนาที่แสวงหาความหมายนอกเหนือศาสนา"],
        "child_rearing": ["ผู้ปกครองที่เน้นความเชื่อฟัง เคารพผู้ใหญ่ และบทบาทครอบครัวแบบดั้งเดิม","ผู้ปกครองที่สมดุลระหว่างวินัยกับการส่งเสริม ให้คุณค่าทั้งความเคารพและความสงสัยใคร่รู้","ผู้ปกครองที่ส่งเสริมความเป็นอิสระและการแสดงออกอย่างสร้างสรรค์","ผู้ปกครองที่ให้คุณค่าการพึ่งพาตนเอง การคิดวิเคราะห์ และการสำรวจอย่างอิสระ"],
        "moral_acceptability": ["ผู้ยึดมั่นในศีลธรรมอย่างเคร่งครัด เชื่อว่าการหย่าร้าง การทำแท้ง ไม่สามารถยอมรับได้","ผู้ระมัดระวังที่ยอมรับบางการกระทำเฉพาะในสถานการณ์รุนแรงเท่านั้น","ผู้ปฏิบัตินิยมที่เชื่อว่าทางเลือกทางศีลธรรมส่วนบุคคลส่วนใหญ่สามารถยอมรับได้ตามบริบท","ผู้เสรีนิยมที่มองว่าการตัดสินใจส่วนบุคคลแทบทั้งหมดเป็นเรื่องของเสรีภาพส่วนบุคคล"],
        "social_trust": ["ผู้ที่ไว้วางใจอย่างลึกซึ้ง เชื่อว่าคนส่วนใหญ่ไว้ใจได้","ผู้ที่ไว้วางใจอย่างระมัดระวัง ไว้ใจเว้นแต่จะมีเหตุผลไม่ให้ไว้ใจ","ผู้คลางแคลงที่ระมัดระวัง เชื่อว่าต้องระวังคนทั่วไป","ผู้ที่ไม่ไว้วางใจอย่างลึกซึ้ง สงสัยว่าคนส่วนใหญ่จะเอาเปรียบถ้ามีโอกาส"],
        "political_participation": ["พลเมืองที่กระตือรือร้นมากในการประท้วง ร้องเรียน และมีส่วนร่วมทางการเมือง","พลเมืองที่มีส่วนร่วมปานกลาง ไปลงคะแนนเสียงสม่ำเสมอและติดตามข่าว","พลเมืองที่สนใจอย่างเฉยๆ ติดตามการเมืองจากระยะไกล","ผู้ที่ไม่สนใจการเมือง ไม่ค่อยไปลงคะแนนเสียงหรือติดตามข่าวการเมือง"],
        "national_pride": ["ผู้รักชาติอย่างแรงกล้าที่ภาคภูมิใจในประวัติศาสตร์และอัตลักษณ์ของชาติ","พลเมืองที่ภูมิใจปานกลาง ให้คุณค่าความสำเร็จของชาติแต่ยอมรับข้อบกพร่อง","ผู้รักชาติที่สงวนท่าที แยกอัตลักษณ์ส่วนตัวจากอัตลักษณ์ชาติ","ผู้มีแนวคิดสากลที่รู้สึกเป็นส่วนหนึ่งของมนุษยชาติมากกว่าชาติใดชาติหนึ่ง"],
        "happiness": ["ผู้ที่มีความพึงพอใจในชีวิตโดยรวมสูงมาก","ผู้ที่รู้สึกพอใจและมีความสุขกับชีวิตโดยทั่วไป","ผู้ที่มีความรู้สึกปนกันเกี่ยวกับคุณภาพชีวิต","ผู้ที่รู้สึกไม่ค่อยพอใจหรือไม่มีความสุขกับสถานการณ์ปัจจุบัน"],
        "gender_equality": ["ผู้สนับสนุนความเสมอภาคอย่างแข็งแกร่ง ยืนยันความเท่าเทียมเต็มที่ระหว่างชายและหญิง","ผู้สนับสนุนความเสมอภาคทางเพศในระดับปานกลาง","ผู้มีแนวโน้มอนุรักษ์ที่ให้คุณค่าบทบาทเพศแบบเสริมกัน","ผู้ยึดมั่นในขนบธรรมเนียมที่เชื่อว่าชายและหญิงมีบทบาทที่แตกต่างกันตามธรรมชาติ"],
        "materialism_orientation": ["ผู้ให้ความสำคัญกับเสรีภาพ การแสดงออก และคุณภาพชีวิตเหนือผลประโยชน์ทางเศรษฐกิจ","ผู้ที่ให้คุณค่าทั้งความเติมเต็มส่วนตัวและความสะดวกสบายทางวัตถุ","ผู้ที่ให้ความสำคัญกับความมั่นคงทางเศรษฐกิจเป็นหลัก","ผู้ที่เน้นการเติบโตทางเศรษฐกิจ ความเป็นระเบียบ และความมั่นคงของชาติเป็นลำดับแรก"],
        "tolerance_diversity": ["ผู้ที่เปิดกว้างอย่างลึกซึ้ง ยินดีต้อนรับเพื่อนบ้านทุกภูมิหลัง","ผู้ที่เปิดกว้างโดยทั่วไปแต่รู้สึกไม่สบายใจกับบางกลุ่ม","ผู้ที่เปิดกว้างอย่างเลือกสรร ชอบเพื่อนบ้านที่คล้ายตนเอง","ผู้ที่ไม่ต้องการอยู่ใกล้กลุ่มนอกหลายประเภท"],
    },
    "el": {
        "religiosity": ["βαθιά πιστός άνθρωπος για τον οποίο η πίστη είναι το κέντρο της καθημερινής ζωής","μετρίως θρησκευόμενος άνθρωπος που πηγαίνει στην εκκλησία και προσεύχεται τακτικά","ονομαστικά θρησκευόμενος άνθρωπος που σπάνια ασκεί τη θρησκεία του","κοσμικός ή μη θρησκευόμενος άνθρωπος που βρίσκει νόημα εκτός οργανωμένης θρησκείας"],
        "child_rearing": ["γονέας που δίνει προτεραιότητα στην υπακοή, τον σεβασμό στην εξουσία και τους παραδοσιακούς οικογενειακούς ρόλους","γονέας που ισορροπεί μεταξύ πειθαρχίας και ενθάρρυνσης","γονέας που καλλιεργεί την ανεξαρτησία και τη δημιουργική έκφραση","γονέας που εκτιμά την αυτάρκεια, την κριτική σκέψη και την ελεύθερη εξερεύνηση"],
        "moral_acceptability": ["αυστηρός ηθικολόγος που θεωρεί ενέργειες όπως το διαζύγιο και η έκτρωση αδικαιολόγητες","συγκρατημένος μετριοπαθής που αποδέχεται κάποιες αμφιλεγόμενες ενέργειες μόνο σε ακραίες περιστάσεις","πραγματιστής φιλελεύθερος που πιστεύει ότι οι περισσότερες προσωπικές ηθικές επιλογές δικαιολογούνται","ελευθεριακός που βλέπει σχεδόν όλες τις προσωπικές αποφάσεις ως θέμα ατομικής ελευθερίας"],
        "social_trust": ["βαθιά εμπιστευτικός άνθρωπος που πιστεύει γενικά ότι οι περισσότεροι άνθρωποι μπορούν να εμπιστευτούν","επιφυλακτικά εμπιστευτικός άνθρωπος","επιφυλακτικός σκεπτικιστής που πιστεύει ότι πρέπει να είσαι προσεκτικός με τους ανθρώπους","βαθιά δύσπιστος άνθρωπος που υποψιάζεται ότι οι περισσότεροι θα εκμεταλλευτούν κάθε ευκαιρία"],
        "political_participation": ["ιδιαίτερα ενεργός πολίτης που διαμαρτύρεται και συμμετέχει σε πολιτική δράση","μετρίως εμπλεκόμενος πολίτης που ψηφίζει τακτικά","παθητικά ενδιαφερόμενος πολίτης που παρακολουθεί την πολιτική από απόσταση","πολιτικά αδιάφορος άνθρωπος που σπάνια ψηφίζει"],
        "national_pride": ["έντονα πατριώτης που νιώθει βαθιά υπερηφάνεια για την ιστορία και ταυτότητα της χώρας","μέτρια υπερήφανος πολίτης","συγκρατημένος πατριώτης που διαχωρίζει την προσωπική από την εθνική ταυτότητα","κοσμοπολίτης που ταυτίζεται περισσότερο με την ανθρωπότητα παρά με ένα έθνος"],
        "happiness": ["άνθρωπος με πολύ υψηλή ικανοποίηση από τη ζωή","άνθρωπος γενικά ικανοποιημένος από τη ζωή","άνθρωπος με ανάμεικτα συναισθήματα για την ποιότητα ζωής","άνθρωπος δυσαρεστημένος με τις τρέχουσες συνθήκες"],
        "gender_equality": ["ισχυρός υποστηρικτής πλήρους ισότητας μεταξύ ανδρών και γυναικών","μέτριος υποστηρικτής ισότητας φύλων","παραδοσιακός που εκτιμά τους συμπληρωματικούς ρόλους φύλων","σταθερός παραδοσιακός που πιστεύει ότι άνδρες και γυναίκες έχουν φυσικά ξεχωριστούς ρόλους"],
        "materialism_orientation": ["μεταϋλιστής που δίνει προτεραιότητα στην ελευθερία και αυτοέκφραση","κλίνει προς μεταϋλισμό εκτιμώντας τόσο προσωπική ολοκλήρωση όσο και υλική άνεση","κλίνει προς υλισμό δίνοντας προτεραιότητα στην οικονομική ασφάλεια","ισχυρός υλιστής που βλέπει την οικονομική ανάπτυξη ως κορυφαία προτεραιότητα"],
        "tolerance_diversity": ["βαθιά ανεκτικός άνθρωπος που καλωσορίζει γείτονες κάθε υπόβαθρου","γενικά ανεκτικός αλλά νιώθει δυσφορία με ορισμένες ομάδες","επιλεκτικά ανεκτικός που προτιμά γείτονες παρόμοιους με τον εαυτό του","αποκλειστικός άνθρωπος που προτιμά να μη ζει κοντά σε πολλές εξωομάδες"],
    },
    "ro": {
        "religiosity": ["credincios profund pentru care credința este centrul vieții zilnice","persoană moderat religioasă care participă la slujbe și se roagă regulat","persoană nominal religioasă care se identifică cu o credință dar practică rar","persoană seculară care găsește sens în afara religiei organizate"],
        "child_rearing": ["părinte care prioritizează ascultarea, respectul față de autoritate și rolurile familiale tradiționale","părinte care echilibrează disciplina cu încurajarea","părinte care cultivă independența și expresia creativă","părinte care valorizează autonomia, gândirea critică și explorarea liberă"],
        "moral_acceptability": ["moralist strict care consideră divorțul, avortul și eutanasia nejustificabile","moderat prudent care acceptă acțiuni controversate doar în circumstanțe extreme","liberal pragmatic care crede că majoritatea alegerilor morale personale sunt justificabile","libertarian permisiv care vede aproape toate deciziile personale ca chestiuni de libertate individuală"],
        "social_trust": ["persoană profund încrezătoare care crede că majoritatea oamenilor pot fi de încredere","persoană prudent încrezătoare","sceptic prudent care crede că trebuie să fii atent cu oamenii","persoană profund neîncrezătoare care suspectează că majoritatea ar profita dacă ar avea ocazia"],
        "political_participation": ["cetățean foarte activ care protestează și se implică în acțiuni politice directe","cetățean moderat implicat care votează regulat","cetățean pasiv interesat care urmărește politica de la distanță","persoană dezinteresată politic care votează sau urmărește știrile politice rar"],
        "national_pride": ["patriot intens care simte o mândrie profundă pentru istoria și identitatea țării","cetățean moderat mândru care valorizează realizările naționale","patriot rezervat care separă identitatea personală de cea națională","cosmopolit care se identifică mai mult cu umanitatea decât cu o singură națiune"],
        "happiness": ["persoană cu satisfacție foarte ridicată față de viață","persoană în general mulțumită de viață","persoană cu sentimente mixte despre calitatea vieții","persoană nemulțumită de circumstanțele actuale"],
        "gender_equality": ["egalitarist puternic care insistă pe egalitatea deplină între bărbați și femei","susținător moderat al egalității de gen","persoană cu înclinații tradiționale care valorizează rolurile complementare de gen","tradiționalist ferm care crede că bărbații și femeile au roluri natural distincte"],
        "materialism_orientation": ["post-materialist care prioritizează libertatea și autoexprimarea","înclinat spre post-materialism valorizând atât împlinirea personală cât și confortul material","înclinat spre materialism prioritizând securitatea economică","materialist puternic care vede creșterea economică ca prioritate supremă"],
        "tolerance_diversity": ["persoană profund incluzivă care primește vecini de orice origine","persoană în general tolerantă dar incomodată de unele grupuri","persoană selectiv tolerantă care preferă vecini similari","persoană exclusivistă care preferă să nu locuiască lângă multe tipuri de grupuri externe"],
    },
    "sr": {
        "religiosity": ["дубоко побожна особа за коју је вера средиште свакодневног живота","умерено религиозна особа која редовно иде у цркву и моли се","номинално религиозна особа која се ретко придржава верских обичаја","секуларна особа која проналази смисао ван организоване религије"],
        "child_rearing": ["родитељ који даје приоритет послушности, поштовању ауторитета и традиционалним породичним улогама","родитељ који балансира дисциплину са подстицањем","родитељ који негује независност и креативно изражавање","родитељ који цени самосталност, критичко мишљење и слободно истраживање"],
        "moral_acceptability": ["строги моралиста који сматра развод, абортус и еутаназију неоправданим","опрезни умерењак који прихвата контроверзне поступке само у екстремним околностима","прагматични либерал који верује да су већина личних моралних избора оправдани","слободоумни либертаријанац који види скоро све личне одлуке као питање индивидуалне слободе"],
        "social_trust": ["особа дубоког поверења која генерално верује да се већини људи може веровати","опрезно поверљива особа","опрезни скептик који верује да треба бити обазрив са људима","дубоко неповерљива особа која сумња да би већина искористила прилику"],
        "political_participation": ["веома активан грађанин који протестује и учествује у политичким акцијама","умерено ангажован грађанин који редовно гласа","пасивно заинтересован грађанин који политику прати издалека","политички незаинтересована особа која ретко гласа"],
        "national_pride": ["интензивни патриота дубоко поносан на историју и идентитет земље","умерено поносан грађанин","уздржани патриота који раздваја лични од националног идентитета","космополита који се идентификује више са човечанством него са једном нацијом"],
        "happiness": ["особа са веома високим задовољством животом","особа генерално задовољна животом","особа са мешовитим осећањима о квалитету живота","особа незадовољна тренутним околностима"],
        "gender_equality": ["снажан заговорник пуне равноправности мушкараца и жена","умерени заговорник родне равноправности","особа традиционалних склоности која вреднује комплементарне родне улоге","чврсти традиционалиста који верује да мушкарци и жене имају природно различите улоге"],
        "materialism_orientation": ["пост-материјалиста који даје приоритет слободи и самоизражавању","нагиње пост-материјализму вреднујући и лично испуњење и материјални комфор","нагиње материјализму дајући приоритет економској сигурности","снажан материјалиста који види економски раст као главни приоритет"],
        "tolerance_diversity": ["дубоко инклузивна особа која прихвата комшије сваког порекла","генерално толерантна особа али се осећа непријатно са неким групама","селективно толерантна особа која преферира сличне комшије","ексклузивна особа која преферира да не живи близу многих типова аутгрупа"],
    },
    "ms": {
        "religiosity": ["penganut yang sangat taat di mana agama adalah pusat kehidupan seharian","orang yang sederhana beragama dan menghadiri ibadah serta berdoa secara tetap","orang yang beragama secara nominal tetapi jarang mengamalkan","orang sekular yang mencari makna di luar agama yang teratur"],
        "child_rearing": ["ibu bapa yang mengutamakan ketaatan, hormat kepada pihak berkuasa dan peranan keluarga tradisional","ibu bapa yang mengimbangi disiplin dengan galakan","ibu bapa yang memupuk kebebasan dan ekspresi kreatif","ibu bapa yang menghargai berdikari, pemikiran kritis dan penerokaan bebas"],
        "moral_acceptability": ["moralis tegas yang menganggap perceraian, pengguguran dan eutanasia tidak boleh dibenarkan","moderat berhati-hati yang menerima tindakan kontroversi hanya dalam keadaan melampau","liberal pragmatik yang percaya kebanyakan pilihan moral peribadi boleh dibenarkan","libertarian permisif yang melihat hampir semua keputusan peribadi sebagai hal kebebasan individu"],
        "social_trust": ["orang yang sangat mempercayai dan percaya kebanyakan orang boleh dipercayai","orang yang berhati-hati dalam mempercayai","skeptik berhati-hati yang percaya perlu berwaspada dengan orang","orang yang sangat tidak mempercayai yang mengesyaki kebanyakan orang akan mengambil kesempatan"],
        "political_participation": ["warganegara sangat aktif yang membantah dan terlibat dalam tindakan politik","warganegara terlibat sederhana yang mengundi secara tetap","warganegara yang berminat secara pasif dan memantau politik dari jauh","orang yang tidak berminat dalam politik dan jarang mengundi"],
        "national_pride": ["patriot yang amat bangga dengan sejarah dan identiti negara","warganegara sederhana bangga yang menghargai pencapaian negara","patriot yang terkawal yang memisahkan identiti peribadi daripada identiti nasional","kosmopolitan yang lebih mengidentifikasikan diri dengan kemanusiaan"],
        "happiness": ["orang yang mempunyai kepuasan hidup keseluruhan yang sangat tinggi","orang yang secara amnya berpuas hati dengan kehidupan","orang yang mempunyai perasaan bercampur tentang kualiti hidup","orang yang agak tidak berpuas hati dengan keadaan semasa"],
        "gender_equality": ["penyokong kuat kesamarataan penuh antara lelaki dan wanita","penyokong sederhana kesamarataan gender","orang yang cenderung tradisional yang menghargai peranan gender yang saling melengkapi","tradisionalis yang percaya lelaki dan wanita mempunyai peranan yang berbeza secara semula jadi"],
        "materialism_orientation": ["pasca-materialis yang mengutamakan kebebasan dan ekspresi diri","cenderung pasca-materialis yang menghargai kepuasan peribadi dan keselesaan material","cenderung materialis yang mengutamakan keselamatan ekonomi","materialis kuat yang melihat pertumbuhan ekonomi sebagai keutamaan"],
        "tolerance_diversity": ["orang yang sangat inklusif dan mengalu-alukan jiran dari sebarang latar belakang","orang yang umumnya toleran tetapi tidak selesa dengan sesetengah kumpulan","orang yang toleran secara terpilih yang lebih suka jiran yang serupa","orang yang eksklusif yang lebih suka tidak tinggal berhampiran banyak kumpulan luar"],
    },
    "bn": {
        "religiosity": ["গভীরভাবে ধর্মপ্রাণ ব্যক্তি যার জন্য বিশ্বাস দৈনন্দিন জীবনের কেন্দ্র","মধ্যম ধর্মপ্রাণ ব্যক্তি যিনি নিয়মিত ইবাদত করেন","নামমাত্র ধর্মপ্রাণ ব্যক্তি যিনি খুব কমই ধর্ম পালন করেন","ধর্মনিরপেক্ষ ব্যক্তি যিনি সংগঠিত ধর্মের বাইরে অর্থ খুঁজে পান"],
        "child_rearing": ["অভিভাবক যিনি আনুগত্য, কর্তৃত্বের প্রতি সম্মান এবং ঐতিহ্যবাহী পারিবারিক ভূমিকাকে অগ্রাধিকার দেন","অভিভাবক যিনি শৃঙ্খলা এবং উৎসাহের মধ্যে ভারসাম্য বজায় রাখেন","অভিভাবক যিনি স্বাধীনতা এবং সৃজনশীল অভিব্যক্তি লালন করেন","অভিভাবক যিনি আত্মনির্ভরশীলতা, সমালোচনামূলক চিন্তাভাবনা এবং মুক্ত অন্বেষণকে সর্বোচ্চ মূল্য দেন"],
        "moral_acceptability": ["কঠোর নীতিবাদী যিনি বিবাহবিচ্ছেদ, গর্ভপাত এবং ইউথানেশিয়াকে কখনই ন্যায়সঙ্গত মনে করেন না","সতর্ক মধ্যপন্থী যিনি শুধুমাত্র চরম পরিস্থিতিতে বিতর্কিত পদক্ষেপ গ্রহণ করেন","বাস্তববাদী উদারপন্থী যিনি বিশ্বাস করেন ব্যক্তিগত নৈতিক পছন্দগুলি ন্যায়সঙ্গত","অনুমতিশীল স্বাধীনতাবাদী যিনি প্রায় সমস্ত ব্যক্তিগত সিদ্ধান্তকে ব্যক্তিগত স্বাধীনতার বিষয় মনে করেন"],
        "social_trust": ["গভীরভাবে বিশ্বাসী ব্যক্তি যিনি মনে করেন বেশিরভাগ মানুষকে বিশ্বাস করা যায়","সতর্কভাবে বিশ্বাসী ব্যক্তি","সতর্ক সংশয়বাদী যিনি মনে করেন মানুষের সাথে সতর্ক থাকা উচিত","গভীরভাবে অবিশ্বাসী ব্যক্তি যিনি সন্দেহ করেন যে বেশিরভাগ মানুষ সুযোগ পেলে শোষণ করবে"],
        "political_participation": ["অত্যন্ত সক্রিয় নাগরিক যিনি প্রতিবাদ করেন এবং রাজনৈতিক কর্মকাণ্ডে অংশগ্রহণ করেন","মধ্যমভাবে জড়িত নাগরিক যিনি নিয়মিত ভোট দেন","নিষ্ক্রিয়ভাবে আগ্রহী নাগরিক যিনি দূর থেকে রাজনীতি অনুসরণ করেন","রাজনৈতিকভাবে উদাসীন ব্যক্তি যিনি খুব কমই ভোট দেন"],
        "national_pride": ["তীব্র দেশপ্রেমিক যিনি দেশের ইতিহাস এবং পরিচয়ে গভীর গর্ব অনুভব করেন","মধ্যমভাবে গর্বিত নাগরিক","সংযত দেশপ্রেমিক যিনি ব্যক্তিগত পরিচয়কে জাতীয় পরিচয় থেকে আলাদা করেন","বিশ্বনাগরিক যিনি যেকোনো একক জাতির চেয়ে মানবতার সাথে বেশি পরিচয় অনুভব করেন"],
        "happiness": ["যার সামগ্রিক জীবন সন্তুষ্টি অত্যন্ত উচ্চ","যিনি সাধারণত জীবনে সন্তুষ্ট এবং খুশি","যার জীবনের মান সম্পর্কে মিশ্র অনুভূতি রয়েছে","যিনি বর্তমান পরিস্থিতিতে কিছুটা অসন্তুষ্ট বা অসুখী"],
        "gender_equality": ["শক্তিশালী সমতাবাদী যিনি সকল ক্ষেত্রে পুরুষ ও নারীর পূর্ণ সমতায় জোর দেন","লিঙ্গ সমতার মধ্যপন্থী সমর্থক","ঐতিহ্যবাহী প্রবণতার ব্যক্তি যিনি পরিপূরক লিঙ্গ ভূমিকাকে মূল্য দেন","দৃঢ় ঐতিহ্যবাদী যিনি বিশ্বাস করেন পুরুষ ও নারীর ভূমিকা প্রকৃতিগতভাবে ভিন্ন"],
        "materialism_orientation": ["উত্তর-বস্তুবাদী যিনি অর্থনৈতিক লাভের চেয়ে স্বাধীনতা এবং আত্মপ্রকাশকে অগ্রাধিকার দেন","উত্তর-বস্তুবাদের দিকে ঝুঁকে যিনি ব্যক্তিগত পরিতৃপ্তি ও বস্তুগত আরাম উভয়ই মূল্য দেন","বস্তুবাদের দিকে ঝুঁকে যিনি অর্থনৈতিক নিরাপত্তাকে অগ্রাধিকার দেন","শক্তিশালী বস্তুবাদী যিনি অর্থনৈতিক প্রবৃদ্ধিকে শীর্ষ অগ্রাধিকার মনে করেন"],
        "tolerance_diversity": ["গভীরভাবে অন্তর্ভুক্তিমূলক ব্যক্তি যিনি যেকোনো পটভূমির প্রতিবেশীকে স্বাগত জানান","সাধারণত সহিষ্ণু কিন্তু কিছু গোষ্ঠীর সাথে অস্বস্তি বোধ করেন","বেছে বেছে সহিষ্ণু যিনি নিজের মতো প্রতিবেশী পছন্দ করেন","বর্জনশীল ব্যক্তি যিনি বিভিন্ন ধরনের বহির্গোষ্ঠীর কাছে থাকতে পছন্দ করেন না"],
    },
    "be": {
        "religiosity": ["глыбока набожны верующы, для якога вера — цэнтр штодзённага жыцця","памяркоўна рэлігійны чалавек, які рэгулярна ходзіць на набажэнствы","намінальна рэлігійны чалавек, які рэдка практыкуе веру","свецкі чалавек, які знаходзіць сэнс паза арганізаванай рэлігіяй"],
        "child_rearing": ["бацька, які аддае перавагу паслухмянасці, павазе да аўтарытэту і традыцыйным сямейным ролям","бацька, які балансуе паміж дысцыплінай і заахвочваннем","бацька, які выхоўвае незалежнасць і творчае самавыражэнне","бацька, які цэніць самастойнасць, крытычнае мысленне і свабоднае даследаванне"],
        "moral_acceptability": ["строгі маралiст, які лiчыць развод, аборт i эўтаназію непапраўднымi","асцярожны памяркоўны, які прымае спрэчныя ўчынкі толькі ў крайніх абставінах","прагматычны ліберал, які верыць, што большасць асабістых маральных выбараў апраўданыя","дазваляльны лібертарыянец, які бачыць амаль усе асабістыя рашэнні як пытанне індывідуальнай свабоды"],
        "social_trust": ["глыбока давяральны чалавек, які верыць, што большасці людзей можна давяраць","асцярожна давяральны чалавек","асцярожны скептык, які лічыць, што з людзьмі трэба быць уважлівым","глыбока недавяральны чалавек, які падазрае, што большасць скарыстаецца нагодай"],
        "political_participation": ["вельмі актыўны грамадзянін, які пратэстуе і ўдзельнічае ў палітычных акцыях","памяркоўна ангажаваны грамадзянін, які рэгулярна галасуе","пасіўна зацікаўлены грамадзянін, які сочыць за палітыкай здалёку","палітычна абыякавы чалавек, які рэдка галасуе"],
        "national_pride": ["інтэнсіўны патрыёт з глыбокім гонарам за гісторыю і ідэнтычнасць краіны","памяркоўна горды грамадзянін","стрыманы патрыёт, які аддзяляе асабістую ідэнтычнасць ад нацыянальнай","касмапаліт, які ідэнтыфікуе сябе больш з чалавецтвам, чым з адной нацыяй"],
        "happiness": ["чалавек з вельмі высокай задаволенасцю жыццём","чалавек, які ў цэлым задаволены жыццём","чалавек са змешанымі пачуццямі адносна якасці жыцця","чалавек, які незадаволены бягучымі абставінамі"],
        "gender_equality": ["моцны эгалітарыст, які настойвае на поўнай роўнасці мужчын і жанчын","памяркоўны прыхільнік гендарнай роўнасці","чалавек з традыцыйнымі схільнасцямі, які цэніць камплементарныя гендарныя ролі","цвёрды традыцыяналіст, які верыць, што мужчыны і жанчыны маюць прыродна розныя ролі"],
        "materialism_orientation": ["пост-матэрыяліст, які аддае перавагу свабодзе і самавыражэнню","схіляецца да пост-матэрыялізму, цэнячы і асабістае задавальненне, і матэрыяльны камфорт","схіляецца да матэрыялізму, аддаючы перавагу эканамічнай бяспецы","моцны матэрыяліст, які бачыць эканамічны рост як галоўны прыярытэт"],
        "tolerance_diversity": ["глыбока інклюзіўны чалавек, які вітае суседзяў любога паходжання","ў цэлым талерантны, але адчувае дыскамфорт з некаторымі групамі","выбарачна талерантны, які аддае перавагу падобным суседзям","эксклюзіўны чалавек, які аддае перавагу не жыць побач з многімі тыпамі аўтгруп"],
    },
}

def validate_batch1_country_codes() -> None:
    """Validate WVS ISO3 formatting and batch-size expectations."""
    if len(BATCH1_WVS_ISO3) != 25:
        raise ValueError(f"batch1 must have 25 countries, got {len(BATCH1_WVS_ISO3)}")
    if len(set(BATCH1_WVS_ISO3)) != len(BATCH1_WVS_ISO3):
        raise ValueError("batch1 has duplicate ISO3 country codes")
    bad = [c for c in BATCH1_WVS_ISO3 if not (len(c) == 3 and c.isupper())]
    if bad:
        raise ValueError(f"batch1 invalid ISO3 format: {bad}")
    missing_lang = [c for c in BATCH1_WVS_ISO3 if c not in BATCH1_COUNTRY_LANG]
    if missing_lang:
        raise ValueError(f"batch1 missing language mapping for: {missing_lang}")


if __name__ == "__main__":
    validate_batch1_country_codes()
    print(f"Batch 1 descriptors: {len(BATCH1_DESCRIPTORS)} languages loaded")
