
======================================================================
You are given data from a Whisper-based Automatic Speech Recognition (ASR) model fine-tuned for Shanghainese (上海话) dialect.

Model Logic:
- Input: Raw audio waveform (Shanghainese speech)
- Encoder: Extracts acoustic features → hidden representations
- SAE Layer: Sparse Autoencoder applied to encoder layer 12 hidden states
- Decoder: Generates text transcription from encoded features
- Output: Chinese text transcription

The provided data describes a **concept** extracted by a Sparse Autoencoder (SAE).
A concept = a speech/linguistic feature detected from the audio that influences transcription output.

Your Task:
Using the character distribution, context windows, and acoustic/linguistic statistics below,
identify **three possible speech/linguistic phenomena** this neuron concept could represent,
ranked in descending order of confidence.

============================================================
DATA FORMAT DEFINITIONS
============================================================

1. **Character Distribution** (主线):
   The most frequent characters (aligned_char) among the top-500 highest-activation tokens.
   Each entry shows: character, occurrence count, mean activation value, and context window examples.
   The context window (aligned_context) is the ±1 character window around the aligned character
   in the transcript, giving local phonetic/lexical context.

2. **Top 30 Token Evidence** (具体证据):
   The 30 individual tokens with highest activation values.
   Each entry shows: activation value, the aligned character, its context window, and the full transcript.
   This preserves per-token detail so you can spot patterns not visible in aggregated statistics.

3. **Acoustic Features**: Audio-level statistics (energy, pitch, spectral centroid, etc.)
   computed over the full transcript of each top-activation token.

4. **Linguistic Features**: Text-level statistics (char count, punctuation, dialect markers, etc.)
   computed over the full transcript of each top-activation token.

============================================================
CURRENT CONCEPT DATA: Neuron #3225
(Total activation events: 36)
============================================================

### Character Distribution (Top 10 chars, from top-500 tokens)
   1. 字=[是]  出现=4次  平均激活=1.193  上下文示例: 拉是三、是呃、搿是，
   2. 字=[光]  出现=3次  平均激活=1.291  上下文示例: 辰光诶
   3. 字=[入]  出现=2次  平均激活=1.383  上下文示例: 入市
   4. 字=[诶]  出现=2次  平均激活=1.313  上下文示例: 诶，、光诶呃
   5. 字=[庭]  出现=2次  平均激活=1.313  上下文示例: 家庭生
   6. 字=[好]  出现=2次  平均激活=1.255  上下文示例: 侬好买
   7. 字=[看]  出现=2次  平均激活=1.191  上下文示例: 好看得
   8. 字=[于]  出现=2次  平均激活=1.194  上下文示例: 至于里
   9. 字=[拉]  出现=2次  平均激活=1.192  上下文示例: 伊拉碰
  10. 字=[伐]  出现=2次  平均激活=1.129  上下文示例: 伐要

### Top 30 Token Evidence (highest activation, per-token detail)
   1. 激活=1.445  字=[入]  上下文=[入市]  整句=[入市埃呃需谨慎，就是搿搿能噶]
   2. 激活=1.440  字=[没]  上下文=[没了]  整句=[没了，现在没了，侪撒地方[+]]
   3. 激活=1.433  字=[带]  上下文=[带小]  整句=[带小拧呃，侬假叫伊穿了哪能样子，伊还伐高兴[+]]
   4. 激活=1.376  字=[诶]  上下文=[诶，]  整句=[诶，全部包好子以后摆辣冰箱里相浸一夜天，然后第二天]
   5. 激活=1.358  字=[庭]  上下文=[家庭生]  整句=[家庭生活呢确实是呃，那个呃小辰光现在三口之家呃，还有么大家庭呢就比较少了]
   6. 激活=1.348  字=[光]  上下文=[辰光诶]  整句=[有辰光诶呃，大吊车还会的打呃，懂伐懂啊]
   7. 激活=1.320  字=[入]  上下文=[入市]  整句=[入市埃呃需谨慎，就是搿搿能噶]
   8. 激活=1.300  字=[好]  上下文=[侬好买]  整句=[侬好买，为撒道理呢]
   9. 激活=1.293  字=[光]  上下文=[辰光诶]  整句=[有辰光诶呃，大吊车还会的打呃，懂伐懂啊]
  10. 激活=1.286  字=[是]  上下文=[拉是三]  整句=[阿拉是三只脚啦，等于侪是隔一条马路葛末，因为为撒道理教育]
  11. 激活=1.268  字=[庭]  上下文=[家庭生]  整句=[家庭生活呢确实是呃，那个呃小辰光现在三口之家呃，还有么大家庭呢就比较少了]
  12. 激活=1.267  字=[呃]  上下文=[拧呃，]  整句=[带小拧呃，侬假叫伊穿了哪能样子，伊还伐高兴[+]]
  13. 激活=1.250  字=[诶]  上下文=[光诶呃]  整句=[有辰光诶呃，大吊车还会的打呃，懂伐懂啊]
  14. 激活=1.250  字=[讲]  上下文=[吾讲侬]  整句=[吾讲侬拔吾，吾伐要，吾拿回去吾伐会的烧，烧伐出搿味道，就是伐好吃]
  15. 激活=1.235  字=[只]  上下文=[三只脚]  整句=[阿拉是三只脚啦，等于侪是隔一条马路葛末，因为为撒道理教育]
  16. 激活=1.233  字=[光]  上下文=[辰光诶]  整句=[有辰光诶呃，大吊车还会的打呃，懂伐懂啊]
  17. 激活=1.229  字=[看]  上下文=[好看得]  整句=[有呃有呃安慰，为撒道理呢吾电视打开好看看，手机打开好，基本上好看得着摸得着]
  18. 激活=1.219  字=[于]  上下文=[至于里]  整句=[至于里向呃比例多少]
  19. 激活=1.218  字=[吃]  上下文=[吃爷]  整句=[吃爷娘呃，伐像阿拉对爷娘侪介好]
  20. 激活=1.211  字=[是]  上下文=[拉是三]  整句=[阿拉是三只脚啦，等于侪是隔一条马路葛末，因为为撒道理教育]
  21. 激活=1.209  字=[好]  上下文=[侬好买]  整句=[侬好买，为撒道理呢]
  22. 激活=1.194  字=[拉]  上下文=[伊拉碰]  整句=[伊拉碰一直碰着事体呃]
  23. 激活=1.190  字=[拉]  上下文=[伊拉碰]  整句=[伊拉碰一直碰着事体呃]
  24. 激活=1.188  字=[侬]  上下文=[侬先]  整句=[侬先叫侬签呃辰光侬慢慢叫看，慢慢叫签，侬讲让吾拍一张下来，吾去问问人家]
  25. 激活=1.185  字=[多]  上下文=[例多少]  整句=[至于里向呃比例多少]
  26. 激活=1.170  字=[于]  上下文=[至于里]  整句=[至于里向呃比例多少]
  27. 激活=1.166  字=[伐]  上下文=[伐要]  整句=[伐要跟侬横查竖查]
  28. 激活=1.164  字=[蹲]  上下文=[蹲了]  整句=[蹲了[PII]的，[PII]的]
  29. 激活=1.153  字=[看]  上下文=[好看得]  整句=[有呃有呃安慰，为撒道理呢吾电视打开好看看，手机打开好，基本上好看得着摸得着]
  30. 激活=1.153  字=[，]  上下文=[诶，脑]  整句=[诶，脑子还蛮灵活呃，还没生了]

### Acoustic Features Statistics (from top-30 transcripts)
  - snr_db: mean=6.754, std=1.041, n=30
  - speech_rate: mean=23.613, std=9.793, n=30
  - pitch_mean: mean=176.153, std=61.741, n=30
  - pitch_std: mean=56.828, std=56.719, n=30
  - pitch_range: mean=203.651, std=154.038, n=30
  - energy_mean: mean=0.049, std=0.040, n=30
  - energy_std: mean=0.049, std=0.040, n=30
  - energy_dynamic_range: mean=41.808, std=7.887, n=30
  - silence_ratio: mean=0.737, std=0.116, n=30
  - speech_duration: mean=0.935, std=0.596, n=30
  - total_duration: mean=3.722, std=1.968, n=30
  - zero_crossing_rate: mean=0.125, std=0.035, n=30
  - spectral_centroid: mean=1674.648, std=349.593, n=30


### Linguistic Features Statistics (from top-30 transcripts)
  - char_count: mean=20.200, std=9.734
  - word_count: mean=1.000, std=0.000
  - contains_numbers: 0/30 (0.0% True)
  - contains_english: 1/30 (3.3% True)
  - contains_punctuation: 24/30 (80.0% True)
  - number_ratio: mean=0.000, std=0.000
  - contains_person_name: 6/30 (20.0% True)
  - contains_place_name: 5/30 (16.7% True)
  - contains_org_name: 1/30 (3.3% True)
  - voiced_initial_ratio: mean=0.405, std=0.168
  - nasal_final_ratio: mean=0.186, std=0.091


============================================================
REQUIRED OUTPUT FORMAT
============================================================

Return a JSON array with exactly 3 hypotheses:

[
  {
    "Reasoning": "<Detailed explanation referencing specific chars, contexts, and acoustic patterns>",
    "Phenomenon": "<Name of the speech/linguistic phenomenon>",
    "Confidence": "<1-4>"
  },
  ...
]

Confidence Scoring:
- 4: Character distribution + context patterns + acoustic trends all strongly align
- 3: Most evidence aligns, minor inconsistencies
- 2: Some evidence aligns, weaker support
- 1: Very little aligns, highly uncertain

============================================================
CHECKLIST FOR REASONING (address each item)
============================================================

1. **Character Pattern Analysis**:
   - Which characters dominate the distribution? Are they phonetically similar?
   - Do the context windows (aligned_context) reveal consistent phonetic neighbors?
   - Are there repeated syllable structures or tonal patterns across contexts?

2. **Context Window Interpretation**:
   - What characters appear before/after the aligned character most often?
   - Do context patterns suggest word boundaries, tone sandhi, or co-articulation effects?

3. **Acoustic Feature Interpretation**:
   - Do energy/pitch patterns suggest specific phonetic categories (voiced/unvoiced, tonal)?
   - Does spectral centroid or zero_crossing_rate indicate fricatives, stops, or nasals?

4. **Dialect-Specific Analysis**:
   - Are the high-frequency characters Shanghainese-specific vocabulary?
   - Do context patterns suggest Wu dialect tone sandhi rules?

5. **Linguistic Level**:
   - Is this a phoneme-level, syllable-level, word-level, or sentence-level feature?

6. **Error Pattern Correlation**:
   - Could this neuron's activation pattern relate to specific ASR confusion types?
   - Homophone errors? Tone errors? Word boundary errors?

======================================================================


======================================================================
You are given data from a Whisper-based Automatic Speech Recognition (ASR) model fine-tuned for Shanghainese (上海话) dialect.

Model Logic:
- Input: Raw audio waveform (Shanghainese speech)
- Encoder: Extracts acoustic features → hidden representations
- SAE Layer: Sparse Autoencoder applied to encoder layer 12 hidden states
- Decoder: Generates text transcription from encoded features
- Output: Chinese text transcription

The provided data describes a **concept** extracted by a Sparse Autoencoder (SAE).
A concept = a speech/linguistic feature detected from the audio that influences transcription output.

Your Task:
Using the character distribution, context windows, and acoustic/linguistic statistics below,
identify **three possible speech/linguistic phenomena** this neuron concept could represent,
ranked in descending order of confidence.

============================================================
DATA FORMAT DEFINITIONS
============================================================

1. **Character Distribution** (主线):
   The most frequent characters (aligned_char) among the top-500 highest-activation tokens.
   Each entry shows: character, occurrence count, mean activation value, and context window examples.
   The context window (aligned_context) is the ±1 character window around the aligned character
   in the transcript, giving local phonetic/lexical context.

2. **Top 30 Token Evidence** (具体证据):
   The 30 individual tokens with highest activation values.
   Each entry shows: activation value, the aligned character, its context window, and the full transcript.
   This preserves per-token detail so you can spot patterns not visible in aggregated statistics.

3. **Acoustic Features**: Audio-level statistics (energy, pitch, spectral centroid, etc.)
   computed over the full transcript of each top-activation token.

4. **Linguistic Features**: Text-level statistics (char count, punctuation, dialect markers, etc.)
   computed over the full transcript of each top-activation token.

============================================================
CURRENT CONCEPT DATA: Neuron #3842
(Total activation events: 4879)
============================================================

### Character Distribution (Top 10 chars, from top-500 tokens)
   1. 字=[搿]  出现=26次  平均激活=2.886  上下文示例: 搿能、是搿能、搿叫、搿记、搿就
   2. 字=[就]  出现=20次  平均激活=2.953  上下文示例: 搿就是、末就是、侬就记、就是、就每
   3. 字=[，]  出现=18次  平均激活=2.885  上下文示例: 哦，东、呃，基、呃，就、哟，吾、喏，搿
   4. 字=[葛]  出现=16次  平均激活=3.184  上下文示例: 房葛末、葛末、，葛末
   5. 字=[北]  出现=15次  平均激活=3.106  上下文示例: 北京
   6. 字=[吾]  出现=14次  平均激活=3.024  上下文示例: 吾就、拔吾，、吾吾晓、吾要、吾对
   7. 字=[是]  出现=14次  平均激活=2.961  上下文示例: 是伐、搿是呃、料是老、拉是，、但是伊
   8. 字=[侬]  出现=12次  平均激活=3.015  上下文示例: 光侬要、讲侬肉、好侬好、为侬，、侬刚
   9. 字=[诶]  出现=12次  平均激活=3.031  上下文示例: 诶就、诶，、，诶还、诶天、，诶些
  10. 字=[乃]  出现=11次  平均激活=3.458  上下文示例: 乃末、，乃末

### Top 30 Token Evidence (highest activation, per-token detail)
   1. 激活=5.812  字=[来]  上下文=[头来一]  整句=[后头来一想是搿种拧，伊讲，吧，就拿搿呃，伊拉，阿拉搿的里，迭呃拧后头做厂长唻]
   2. 激活=5.717  字=[小]  上下文=[带小拧]  整句=[带小拧呃，侬假叫伊穿了哪能样子，伊还伐高兴[+]]
   3. 激活=5.715  字=[带]  上下文=[带小]  整句=[带小拧呃，侬假叫伊穿了哪能样子，伊还伐高兴[+]]
   4. 激活=5.661  字=[咖]  上下文=[咖喱]  整句=[咖喱嘛，就倒眼咖喱粉]
   5. 激活=5.488  字=[乃]  上下文=[乃末]  整句=[乃末诶呃叫撒啊]
   6. 激活=5.476  字=[就]  上下文=[搿就是]  整句=[搿就是服务呃，星级服务呃搿呃，标准上去了呀。]
   7. 激活=5.344  字=[乃]  上下文=[乃末]  整句=[乃末喃，迭呃叫撒啊]
   8. 激活=5.332  字=[等]  上下文=[等等]  整句=[等等，还有叫撒]
   9. 激活=5.182  字=[眼]  上下文=[一眼，]  整句=[再大一眼，诶呦，伐要侬了，蛮叫学堂里开会咯撒叫伊拉爷娘去，还伐要侬搿呃老头老太去唻，吾讲呃]
  10. 激活=5.115  字=[阿]  上下文=[嗯阿拉]  整句=[嗯阿拉阿拉阿拉喃就平常，因为是做品牌呃啦]
  11. 激活=5.019  字=[嗯]  上下文=[嗯商]  整句=[嗯商业经济呢也就是讲到搿呃]
  12. 激活=4.968  字=[喏]  上下文=[喏，]  整句=[喏，搿能噶呢就讲交流起来方便点有撒事体照顾得到]
  13. 激活=4.965  字=[吾]  上下文=[吾就]  整句=[吾就觉着伊有搿一点]
  14. 激活=4.904  字=[所]  上下文=[所以]  整句=[所以就是讲]
  15. 激活=4.878  字=[。]  上下文=[啦。现]  整句=[对伐啦。现在讲起来伐稀奇唻，现在册那假使要调查侬老便当呃，一只手机，一只电脑马上就甩过去了]
  16. 激活=4.878  字=[葛]  上下文=[房葛末]  整句=[啊里几间包房葛末就拔伊订下来，侬假使讲去了晚呃闲话呢，年夜饭就订伐着了。]
  17. 激活=4.848  字=[搿]  上下文=[搿能]  整句=[搿能噶喃，就对得起自家]
  18. 激活=4.772  字=[假]  上下文=[假使]  整句=[假使吾打了，吾还进去了]
  19. 激活=4.742  字=[侬]  上下文=[光侬要]  整句=[到辰光侬要想订呃辰光，哎呦自家心仪呃饭店就好了，机会侪没了]
  20. 激活=4.716  字=[关]  上下文=[搿关系]  整句=[搿关系呢就讲大家侪是讲老巧妙呃，其实上侪老巧妙大家要一道要想好]
  21. 激活=4.563  字=[所]  上下文=[所以]  整句=[所以有种经营勒好呃饭店哦，侬看伊呃客座率老是老高呃]
  22. 激活=4.554  字=[，]  上下文=[哦，东]  整句=[哦，东昌斑呃，东昌斑就是鲳鱼啊[+]]
  23. 激活=4.542  字=[以]  上下文=[所以有]  整句=[所以有辰光要叫阿拉带伊拉带小拧咯撒，伊小拧的话，第三代呃小拧呐也蛮难带呃]
  24. 激活=4.502  字=[葛]  上下文=[葛末]  整句=[葛末空辣海么就就没收益嘞，伊拉就会得想出了搿办法]
  25. 激活=4.460  字=[就]  上下文=[末就是]  整句=[嗯，葛末就是讲，嗯，搿呃喃能就每天要吃水果，乃界喃就是讲顶好吃两只]
  26. 激活=4.447  字=[拉]  上下文=[阿拉代]  整句=[乃末跑到外头去喃，阿拉代表阿拉阿爸爸搿一房呢，阿拉屋里相呃兄弟姐妹阿拉爸爸呃子女呢侪要团结好]
  27. 激活=4.434  字=[嗯]  上下文=[嗯，]  整句=[嗯，葛末就是讲，嗯，搿呃喃能就每天要吃水果，乃界喃就是讲顶好吃两只]
  28. 激活=4.427  字=[好]  上下文=[老好呃]  整句=[老好呃一种撒啊，就是讲网哴向搿种，就是搿性价比哦，吾就觉着老好呃]
  29. 激活=4.390  字=[喏]  上下文=[喏，]  整句=[喏，搿能噶呢就讲交流起来方便点有撒事体照顾得到]
  30. 激活=4.388  字=[油]  上下文=[样油水]  整句=[然后这样油水都煮掉了，再蒸的时候油全被梅干菜吸收]

### Acoustic Features Statistics (from top-30 transcripts)
  - snr_db: mean=5.718, std=0.915, n=30
  - speech_rate: mean=17.977, std=6.749, n=30
  - pitch_mean: mean=207.893, std=35.496, n=30
  - pitch_std: mean=47.002, std=24.938, n=30
  - pitch_range: mean=202.240, std=121.643, n=30
  - energy_mean: mean=0.028, std=0.018, n=30
  - energy_std: mean=0.023, std=0.014, n=30
  - energy_dynamic_range: mean=40.629, std=5.912, n=30
  - silence_ratio: mean=0.679, std=0.114, n=30
  - speech_duration: mean=1.308, std=0.636, n=30
  - total_duration: mean=4.371, std=2.126, n=30
  - zero_crossing_rate: mean=0.154, std=0.039, n=30
  - spectral_centroid: mean=1969.204, std=332.665, n=30


### Linguistic Features Statistics (from top-30 transcripts)
  - char_count: mean=23.767, std=11.904
  - word_count: mean=1.000, std=0.000
  - contains_numbers: 0/30 (0.0% True)
  - contains_english: 0/30 (0.0% True)
  - contains_punctuation: 26/30 (86.7% True)
  - number_ratio: mean=0.000, std=0.000
  - contains_person_name: 4/30 (13.3% True)
  - contains_place_name: 0/30 (0.0% True)
  - contains_org_name: 0/30 (0.0% True)
  - voiced_initial_ratio: mean=0.431, std=0.109
  - nasal_final_ratio: mean=0.162, std=0.097


============================================================
REQUIRED OUTPUT FORMAT
============================================================

Return a JSON array with exactly 3 hypotheses:

[
  {
    "Reasoning": "<Detailed explanation referencing specific chars, contexts, and acoustic patterns>",
    "Phenomenon": "<Name of the speech/linguistic phenomenon>",
    "Confidence": "<1-4>"
  },
  ...
]

Confidence Scoring:
- 4: Character distribution + context patterns + acoustic trends all strongly align
- 3: Most evidence aligns, minor inconsistencies
- 2: Some evidence aligns, weaker support
- 1: Very little aligns, highly uncertain

============================================================
CHECKLIST FOR REASONING (address each item)
============================================================

1. **Character Pattern Analysis**:
   - Which characters dominate the distribution? Are they phonetically similar?
   - Do the context windows (aligned_context) reveal consistent phonetic neighbors?
   - Are there repeated syllable structures or tonal patterns across contexts?

2. **Context Window Interpretation**:
   - What characters appear before/after the aligned character most often?
   - Do context patterns suggest word boundaries, tone sandhi, or co-articulation effects?

3. **Acoustic Feature Interpretation**:
   - Do energy/pitch patterns suggest specific phonetic categories (voiced/unvoiced, tonal)?
   - Does spectral centroid or zero_crossing_rate indicate fricatives, stops, or nasals?

4. **Dialect-Specific Analysis**:
   - Are the high-frequency characters Shanghainese-specific vocabulary?
   - Do context patterns suggest Wu dialect tone sandhi rules?

5. **Linguistic Level**:
   - Is this a phoneme-level, syllable-level, word-level, or sentence-level feature?

6. **Error Pattern Correlation**:
   - Could this neuron's activation pattern relate to specific ASR confusion types?
   - Homophone errors? Tone errors? Word boundary errors?

======================================================================


======================================================================
You are given data from a Whisper-based Automatic Speech Recognition (ASR) model fine-tuned for Shanghainese (上海话) dialect.

Model Logic:
- Input: Raw audio waveform (Shanghainese speech)
- Encoder: Extracts acoustic features → hidden representations
- SAE Layer: Sparse Autoencoder applied to encoder layer 12 hidden states
- Decoder: Generates text transcription from encoded features
- Output: Chinese text transcription

The provided data describes a **concept** extracted by a Sparse Autoencoder (SAE).
A concept = a speech/linguistic feature detected from the audio that influences transcription output.

Your Task:
Using the character distribution, context windows, and acoustic/linguistic statistics below,
identify **three possible speech/linguistic phenomena** this neuron concept could represent,
ranked in descending order of confidence.

============================================================
DATA FORMAT DEFINITIONS
============================================================

1. **Character Distribution** (主线):
   The most frequent characters (aligned_char) among the top-500 highest-activation tokens.
   Each entry shows: character, occurrence count, mean activation value, and context window examples.
   The context window (aligned_context) is the ±1 character window around the aligned character
   in the transcript, giving local phonetic/lexical context.

2. **Top 30 Token Evidence** (具体证据):
   The 30 individual tokens with highest activation values.
   Each entry shows: activation value, the aligned character, its context window, and the full transcript.
   This preserves per-token detail so you can spot patterns not visible in aggregated statistics.

3. **Acoustic Features**: Audio-level statistics (energy, pitch, spectral centroid, etc.)
   computed over the full transcript of each top-activation token.

4. **Linguistic Features**: Text-level statistics (char count, punctuation, dialect markers, etc.)
   computed over the full transcript of each top-activation token.

============================================================
CURRENT CONCEPT DATA: Neuron #5929
(Total activation events: 1474)
============================================================

### Character Distribution (Top 10 chars, from top-500 tokens)
   1. 字=[，]  出现=30次  平均激活=3.294  上下文示例: 搿，样、哦，每、呐，呃、喃，阿、子，十
   2. 字=[是]  出现=22次  平均激活=3.703  上下文示例: 就是，、妇是侬、讲是呃、是呃、侪是那
   3. 字=[葛]  出现=22次  平均激活=3.167  上下文示例: 葛末
   4. 字=[末]  出现=18次  平均激活=3.034  上下文示例: 葛末就、乃末呢、葛末另、葛末阿、葛末喏
   5. 字=[侬]  出现=17次  平均激活=3.431  上下文示例: 侬伐、伐侬现、到侬现、侬再、侬先
   6. 字=[伐]  出现=15次  平均激活=2.988  上下文示例: 对伐，、伐弄、对伐啦、伐阁、伐要
   7. 字=[搿]  出现=13次  平均激活=3.004  上下文示例: 搿房、搿讲、搿关、是搿能、搿能
   8. 字=[有]  出现=11次  平均激活=3.463  上下文示例: 家有种、还有交、呢有点、拧有种、有辰
   9. 字=[就]  出现=9次  平均激活=3.118  上下文示例: 侬就记、就是、呢就是、勒就是、拉就没
  10. 字=[吾]  出现=8次  平均激活=3.562  上下文示例: 吾要、末吾现、为吾做、吾现、吾就

### Top 30 Token Evidence (highest activation, per-token detail)
   1. 激活=6.970  字=[拧]  上下文=[小拧有]  整句=[葛末小拧有种小拧嘛，呃肯读书呃，有种小拧伐肯读书呃？伐肯读书嘛]
   2. 激活=6.912  字=[是]  上下文=[就是，]  整句=[就是，去搿能噶样子对伐好像性价比对伐啦，也老高呃]
   3. 激活=6.855  字=[葛]  上下文=[葛末]  整句=[葛末就选一般性呃]
   4. 激活=6.755  字=[撒]  上下文=[得撒地]  整句=[呃，吾吾晓得撒地方好呃饭店，阿拉也要交流交流，葛末就还是要到好呃地方去]
   5. 激活=6.499  字=[呢]  上下文=[拧呢，]  整句=[人家有种拧呢，卖特房子，正好有种是为了出出去，出去正好有笔资金]
   6. 激活=6.494  字=[饭]  上下文=[夜饭喃]  整句=[订年夜饭喃，一般伊来了中秋节前后就开始要订了啦，订了以后]
   7. 激活=6.439  字=[吾]  上下文=[吾要]  整句=[吾要，情愿巨眼]
   8. 激活=6.327  字=[就]  上下文=[侬就记]  整句=[而且侬就记性要好，侬就讲，哦，吾后头再去，经理，还能够拿搿桩事体]
   9. 激活=6.299  字=[中]  上下文=[中，]  整句=[中，中翅]
  10. 激活=6.223  字=[几]  上下文=[几呃]  整句=[几呃，朋友对伐，现在跟吾讲，唉到蛮好呃]
  11. 激活=6.145  字=[确]  上下文=[呢确实]  整句=[家庭生活呢确实是呃，那个呃小辰光现在三口之家呃，还有么大家庭呢就比较少了]
  12. 激活=6.073  字=[入]  上下文=[入市]  整句=[入市埃呃需谨慎，就是搿搿能噶]
  13. 激活=6.034  字=[穿]  上下文=[穿勒]  整句=[穿勒就是讲要得体，就是讲穿呃符合侬搿性格]
  14. 激活=5.990  字=[是]  上下文=[妇是侬]  整句=[新妇是侬侬儿子欢喜呃拧，新妇是侬孙子呃]
  15. 激活=5.936  字=[侬]  上下文=[侬伐]  整句=[侬伐要对孙子，拿孙子当宝贝，好像孙子是侬撒拧撒拧]
  16. 激活=5.871  字=[办]  上下文=[想办法]  整句=[就是讲要想办法要轻松。葛末轻松呃，对自家来讲是是也是一种享受对伐]
  17. 激活=5.721  字=[点]  上下文=[点到]  整句=[点到为止对媳妇是点到为止，伐好多讲]
  18. 激活=5.564  字=[有]  上下文=[家有种]  整句=[人家有种拧呢，卖特房子，正好有种是为了出出去，出去正好有笔资金]
  19. 激活=5.537  字=[入]  上下文=[入市]  整句=[入市埃呃需谨慎，就是搿搿能噶]
  20. 激活=5.520  字=[夜]  上下文=[年夜饭]  整句=[订年夜饭喃，一般伊来了中秋节前后就开始要订了啦，订了以后]
  21. 激活=5.490  字=[如]  上下文=[如果]  整句=[如果讲搿只饭店拧流量]
  22. 激活=5.415  字=[好]  上下文=[勒好，]  整句=[侬吃勒好，穿勒好，葛末搿能噶喃，就自家喃有得自信心]
  23. 激活=5.393  字=[是]  上下文=[讲是呃]  整句=[应该讲是呃伐大会的埃呃，因为伊拉开到现在几十年唻]
  24. 激活=5.355  字=[分]  上下文=[是分开]  整句=[吃饭辰光呢就是分开来吃了，老早点呢介许多拧吃，觉着老开心呃，只圆台面吃好子埃呃]
  25. 激活=5.338  字=[电]  上下文=[开电梯]  整句=[开电梯还要考诶，要有证书]
  26. 激活=5.337  字=[，]  上下文=[搿，样]  整句=[搿，样样伐吃伊营养丰，全面呃]
  27. 激活=5.306  字=[就]  上下文=[侬就记]  整句=[而且侬就记性要好，侬就讲，哦，吾后头再去，经理，还能够拿搿桩事体]
  28. 激活=5.281  字=[是]  上下文=[是呃]  整句=[是呃呀，拔那正好拔那]
  29. 激活=5.275  字=[侬]  上下文=[伐侬现]  整句=[对伐侬现在查一查啊，对伐侬讲老拧寻呃厕所啊，高头侪好显示出来，饭店啊]
  30. 激活=5.265  字=[起]  上下文=[最起码]  整句=[最最起码做基金要一百万资金，葛末阿拉呢，心里相也辣想，阿拉也坏特过，毕竟也工薪家庭]

### Acoustic Features Statistics (from top-30 transcripts)
  - snr_db: mean=5.661, std=1.100, n=30
  - speech_rate: mean=22.767, std=15.134, n=30
  - pitch_mean: mean=189.049, std=38.199, n=30
  - pitch_std: mean=45.731, std=20.964, n=30
  - pitch_range: mean=194.171, std=93.495, n=30
  - energy_mean: mean=0.030, std=0.016, n=30
  - energy_std: mean=0.028, std=0.015, n=30
  - energy_dynamic_range: mean=42.172, std=6.771, n=30
  - silence_ratio: mean=0.755, std=0.086, n=30
  - speech_duration: mean=1.079, std=0.548, n=30
  - total_duration: mean=4.418, std=1.789, n=30
  - zero_crossing_rate: mean=0.188, std=0.044, n=30
  - spectral_centroid: mean=2222.731, std=365.295, n=30


### Linguistic Features Statistics (from top-30 transcripts)
  - char_count: mean=23.167, std=10.208
  - word_count: mean=1.000, std=0.000
  - contains_numbers: 0/30 (0.0% True)
  - contains_english: 0/30 (0.0% True)
  - contains_punctuation: 28/30 (93.3% True)
  - number_ratio: mean=0.000, std=0.000
  - contains_person_name: 4/30 (13.3% True)
  - contains_place_name: 2/30 (6.7% True)
  - contains_org_name: 0/30 (0.0% True)
  - voiced_initial_ratio: mean=0.414, std=0.158
  - nasal_final_ratio: mean=0.211, std=0.119


============================================================
REQUIRED OUTPUT FORMAT
============================================================

Return a JSON array with exactly 3 hypotheses:

[
  {
    "Reasoning": "<Detailed explanation referencing specific chars, contexts, and acoustic patterns>",
    "Phenomenon": "<Name of the speech/linguistic phenomenon>",
    "Confidence": "<1-4>"
  },
  ...
]

Confidence Scoring:
- 4: Character distribution + context patterns + acoustic trends all strongly align
- 3: Most evidence aligns, minor inconsistencies
- 2: Some evidence aligns, weaker support
- 1: Very little aligns, highly uncertain

============================================================
CHECKLIST FOR REASONING (address each item)
============================================================

1. **Character Pattern Analysis**:
   - Which characters dominate the distribution? Are they phonetically similar?
   - Do the context windows (aligned_context) reveal consistent phonetic neighbors?
   - Are there repeated syllable structures or tonal patterns across contexts?

2. **Context Window Interpretation**:
   - What characters appear before/after the aligned character most often?
   - Do context patterns suggest word boundaries, tone sandhi, or co-articulation effects?

3. **Acoustic Feature Interpretation**:
   - Do energy/pitch patterns suggest specific phonetic categories (voiced/unvoiced, tonal)?
   - Does spectral centroid or zero_crossing_rate indicate fricatives, stops, or nasals?

4. **Dialect-Specific Analysis**:
   - Are the high-frequency characters Shanghainese-specific vocabulary?
   - Do context patterns suggest Wu dialect tone sandhi rules?

5. **Linguistic Level**:
   - Is this a phoneme-level, syllable-level, word-level, or sentence-level feature?

6. **Error Pattern Correlation**:
   - Could this neuron's activation pattern relate to specific ASR confusion types?
   - Homophone errors? Tone errors? Word boundary errors?

======================================================================


======================================================================
You are given data from a Whisper-based Automatic Speech Recognition (ASR) model fine-tuned for Shanghainese (上海话) dialect.

Model Logic:
- Input: Raw audio waveform (Shanghainese speech)
- Encoder: Extracts acoustic features → hidden representations
- SAE Layer: Sparse Autoencoder applied to encoder layer 12 hidden states
- Decoder: Generates text transcription from encoded features
- Output: Chinese text transcription

The provided data describes a **concept** extracted by a Sparse Autoencoder (SAE).
A concept = a speech/linguistic feature detected from the audio that influences transcription output.

Your Task:
Using the character distribution, context windows, and acoustic/linguistic statistics below,
identify **three possible speech/linguistic phenomena** this neuron concept could represent,
ranked in descending order of confidence.

============================================================
DATA FORMAT DEFINITIONS
============================================================

1. **Character Distribution** (主线):
   The most frequent characters (aligned_char) among the top-500 highest-activation tokens.
   Each entry shows: character, occurrence count, mean activation value, and context window examples.
   The context window (aligned_context) is the ±1 character window around the aligned character
   in the transcript, giving local phonetic/lexical context.

2. **Top 30 Token Evidence** (具体证据):
   The 30 individual tokens with highest activation values.
   Each entry shows: activation value, the aligned character, its context window, and the full transcript.
   This preserves per-token detail so you can spot patterns not visible in aggregated statistics.

3. **Acoustic Features**: Audio-level statistics (energy, pitch, spectral centroid, etc.)
   computed over the full transcript of each top-activation token.

4. **Linguistic Features**: Text-level statistics (char count, punctuation, dialect markers, etc.)
   computed over the full transcript of each top-activation token.

============================================================
CURRENT CONCEPT DATA: Neuron #1291
(Total activation events: 248)
============================================================

### Character Distribution (Top 10 chars, from top-500 tokens)
   1. 字=[伐]  出现=15次  平均激活=1.448  上下文示例: 对伐，、相伐好、对伐要、对伐侬、伐好
   2. 字=[葛]  出现=13次  平均激活=1.299  上下文示例: 葛末
   3. 字=[伊]  出现=12次  平均激活=1.399  上下文示例: 伊拉、到伊、伊拿、伊好、伊没
   4. 字=[搿]  出现=12次  平均激活=1.248  上下文示例: 搿呃、搿是、呃搿搿、搿胡、搿房
   5. 字=[老]  出现=9次  平均激活=1.335  上下文示例: 拉老夫、末老两、老开
   6. 字=[跌]  出现=9次  平均激活=1.421  上下文示例: 跌到、跌
   7. 字=[呃]  出现=9次  平均激活=1.253  上下文示例: 呃搿、呃，、呃化、撒呃，、对呃呀
   8. 字=[末]  出现=7次  平均激活=1.411  上下文示例: 乃末养、葛末，、葛末现、葛末阿
   9. 字=[对]  出现=7次  平均激活=1.193  上下文示例: 对伐、对呃
  10. 字=[六]  出现=6次  平均激活=1.546  上下文示例: 六十、六月

### Top 30 Token Evidence (highest activation, per-token detail)
   1. 激活=2.106  字=[伊]  上下文=[伊拉]  整句=[伊拉也蛮残，侪做节目咯撒]
   2. 激活=1.935  字=[者]  上下文=[或者下]  整句=[进去，或者下半日就拿出来，也有呃，或者明朝拿出来，也有呃，伊搿只产品只有只利率低]
   3. 激活=1.909  字=[头]  上下文=[花头]  整句=[格勒一天到夜帮伊翻花头]
   4. 激活=1.879  字=[六]  上下文=[六十]  整句=[六十六]
   5. 激活=1.840  字=[老]  上下文=[拉老夫]  整句=[葛末阿拉老夫妻两呃呢就等辣屋里相享受自家呃生活但是呢因为]
   6. 激活=1.831  字=[辣]  上下文=[辣辣]  整句=[辣辣是，一二年辰光伐是，一般性人家是，伐是做股票呃，侪进去孛相嘛]
   7. 激活=1.830  字=[跌]  上下文=[跌到]  整句=[跌到现在诶，就是呢，跌到现在，撒道理伊从五百万，跌到现在只有一百多万了]
   8. 激活=1.830  字=[格]  上下文=[格勒]  整句=[格勒一天到夜帮伊翻花头]
   9. 激活=1.825  字=[头]  上下文=[花头]  整句=[格勒一天到夜帮伊翻花头]
  10. 激活=1.800  字=[[]  上下文=[[L]  整句=[[LAUGHTER]]
  11. 激活=1.792  字=[头]  上下文=[花头]  整句=[格勒一天到夜帮伊翻花头]
  12. 激活=1.778  字=[伐]  上下文=[对伐，]  整句=[对伐，搿上有呃爷爷奶奶咯对伐，老呃四呃对伐，再爷娘好唻六呃拧宠老伊]
  13. 激活=1.774  字=[格]  上下文=[格勒]  整句=[格勒一天到夜帮伊翻花头]
  14. 激活=1.756  字=[六]  上下文=[六月]  整句=[六月十三号，六月十七号出来呃]
  15. 激活=1.724  字=[阿]  上下文=[阿拉]  整句=[阿拉老早喏，自家屋里相弄了多少清爽，侬现在弄了搿能样子，葛末人家又没叫侬来打扫咯，侬觉着乱七八糟，人家物事]
  16. 激活=1.709  字=[伐]  上下文=[相伐好]  整句=[葛末卖相伐好看，吃呃感觉还就伐一样，搿碧绿生青多少好辣，侬讲]
  17. 激活=1.693  字=[葛]  上下文=[葛末]  整句=[葛末，后头文化大革命唻]
  18. 激活=1.690  字=[伊]  上下文=[伊拉]  整句=[伊拉也蛮残，侪做节目咯撒]
  19. 激活=1.687  字=[诶]  上下文=[诶，]  整句=[诶，真呃老节约伊讲，伊拉埃面外头撒呃奶茶咯，撒呃，诶，撒呃，诶种噶撒呃]
  20. 激活=1.684  字=[搿]  上下文=[搿呃]  整句=[搿呃物事吾侪伐懂诶]
  21. 激活=1.684  字=[里]  上下文=[里呃]  整句=[里呃银行里哦]
  22. 激活=1.663  字=[带]  上下文=[带小]  整句=[带小拧呃，侬假叫伊穿了哪能样子，伊还伐高兴[+]]
  23. 激活=1.658  字=[末]  上下文=[乃末养]  整句=[乃末养鳄鱼也有呃，搿是伐对了对伐，搿侪是凶猛动物要伤拧诶]
  24. 激活=1.649  字=[葛]  上下文=[葛末]  整句=[葛末阿拉老夫妻两呃呢就等辣屋里相享受自家呃生活但是呢因为]
  25. 激活=1.646  字=[是]  上下文=[就是讲]  整句=[就是讲要想办法要轻松。葛末轻松呃，对自家来讲是是也是一种享受对伐]
  26. 激活=1.636  字=[个]  上下文=[整个上]  整句=[整个上海伐是侪好走了嘛，阿拉是，喏，喏，现现在讲起来莘庄]
  27. 激活=1.631  字=[里]  上下文=[里呃]  整句=[里呃银行里哦]
  28. 激活=1.629  字=[搿]  上下文=[搿呃]  整句=[搿呃物事吾侪伐懂诶]
  29. 激活=1.623  字=[伐]  上下文=[对伐要]  整句=[对伐要养老唻对伐，现在埃呃阿拉搿种退休工资到养老呃搿呃好呃基地呢是进伐去呃]
  30. 激活=1.623  字=[者]  上下文=[或者下]  整句=[进去，或者下半日就拿出来，也有呃，或者明朝拿出来，也有呃，伊搿只产品只有只利率低]

### Acoustic Features Statistics (from top-30 transcripts)
  - snr_db: mean=6.199, std=1.130, n=30
  - speech_rate: mean=17.407, std=8.826, n=30
  - pitch_mean: mean=159.459, std=50.305, n=30
  - pitch_std: mean=37.816, std=26.414, n=30
  - pitch_range: mean=146.890, std=85.714, n=30
  - energy_mean: mean=0.028, std=0.022, n=30
  - energy_std: mean=0.025, std=0.021, n=30
  - energy_dynamic_range: mean=38.680, std=7.974, n=30
  - silence_ratio: mean=0.650, std=0.113, n=30
  - speech_duration: mean=1.228, std=0.657, n=30
  - total_duration: mean=3.963, std=2.237, n=30
  - zero_crossing_rate: mean=0.143, std=0.045, n=30
  - spectral_centroid: mean=1871.996, std=388.331, n=30


### Linguistic Features Statistics (from top-30 transcripts)
  - char_count: mean=21.667, std=12.970
  - word_count: mean=1.000, std=0.000
  - contains_numbers: 0/30 (0.0% True)
  - contains_english: 1/30 (3.3% True)
  - contains_punctuation: 17/30 (56.7% True)
  - number_ratio: mean=0.000, std=0.000
  - contains_person_name: 1/30 (3.3% True)
  - contains_place_name: 1/30 (3.3% True)
  - contains_org_name: 2/30 (6.7% True)
  - voiced_initial_ratio: mean=0.416, std=0.230
  - nasal_final_ratio: mean=0.134, std=0.099


============================================================
REQUIRED OUTPUT FORMAT
============================================================

Return a JSON array with exactly 3 hypotheses:

[
  {
    "Reasoning": "<Detailed explanation referencing specific chars, contexts, and acoustic patterns>",
    "Phenomenon": "<Name of the speech/linguistic phenomenon>",
    "Confidence": "<1-4>"
  },
  ...
]

Confidence Scoring:
- 4: Character distribution + context patterns + acoustic trends all strongly align
- 3: Most evidence aligns, minor inconsistencies
- 2: Some evidence aligns, weaker support
- 1: Very little aligns, highly uncertain

============================================================
CHECKLIST FOR REASONING (address each item)
============================================================

1. **Character Pattern Analysis**:
   - Which characters dominate the distribution? Are they phonetically similar?
   - Do the context windows (aligned_context) reveal consistent phonetic neighbors?
   - Are there repeated syllable structures or tonal patterns across contexts?

2. **Context Window Interpretation**:
   - What characters appear before/after the aligned character most often?
   - Do context patterns suggest word boundaries, tone sandhi, or co-articulation effects?

3. **Acoustic Feature Interpretation**:
   - Do energy/pitch patterns suggest specific phonetic categories (voiced/unvoiced, tonal)?
   - Does spectral centroid or zero_crossing_rate indicate fricatives, stops, or nasals?

4. **Dialect-Specific Analysis**:
   - Are the high-frequency characters Shanghainese-specific vocabulary?
   - Do context patterns suggest Wu dialect tone sandhi rules?

5. **Linguistic Level**:
   - Is this a phoneme-level, syllable-level, word-level, or sentence-level feature?

6. **Error Pattern Correlation**:
   - Could this neuron's activation pattern relate to specific ASR confusion types?
   - Homophone errors? Tone errors? Word boundary errors?

======================================================================


======================================================================
You are given data from a Whisper-based Automatic Speech Recognition (ASR) model fine-tuned for Shanghainese (上海话) dialect.

Model Logic:
- Input: Raw audio waveform (Shanghainese speech)
- Encoder: Extracts acoustic features → hidden representations
- SAE Layer: Sparse Autoencoder applied to encoder layer 12 hidden states
- Decoder: Generates text transcription from encoded features
- Output: Chinese text transcription

The provided data describes a **concept** extracted by a Sparse Autoencoder (SAE).
A concept = a speech/linguistic feature detected from the audio that influences transcription output.

Your Task:
Using the character distribution, context windows, and acoustic/linguistic statistics below,
identify **three possible speech/linguistic phenomena** this neuron concept could represent,
ranked in descending order of confidence.

============================================================
DATA FORMAT DEFINITIONS
============================================================

1. **Character Distribution** (主线):
   The most frequent characters (aligned_char) among the top-500 highest-activation tokens.
   Each entry shows: character, occurrence count, mean activation value, and context window examples.
   The context window (aligned_context) is the ±1 character window around the aligned character
   in the transcript, giving local phonetic/lexical context.

2. **Top 30 Token Evidence** (具体证据):
   The 30 individual tokens with highest activation values.
   Each entry shows: activation value, the aligned character, its context window, and the full transcript.
   This preserves per-token detail so you can spot patterns not visible in aggregated statistics.

3. **Acoustic Features**: Audio-level statistics (energy, pitch, spectral centroid, etc.)
   computed over the full transcript of each top-activation token.

4. **Linguistic Features**: Text-level statistics (char count, punctuation, dialect markers, etc.)
   computed over the full transcript of each top-activation token.

============================================================
CURRENT CONCEPT DATA: Neuron #4300
(Total activation events: 1495)
============================================================

### Character Distribution (Top 10 chars, from top-500 tokens)
   1. 字=[，]  出现=22次  平均激活=1.829  上下文示例: 嗯，有、是，像、慰，为、价，诶、讲，嗯
   2. 字=[呃]  出现=20次  平均激活=2.053  上下文示例: 呃两、性呃撒、末呃一、是呃，、烧呃搿
   3. 字=[搿]  出现=16次  平均激活=1.904  上下文示例: 照搿方、搿能、侬搿，、拿搿搿、侬搿有
   4. 字=[吾]  出现=16次  平均激活=1.964  上下文示例: ，吾吾、吾高、吾讲、咯吾基、吾就
   5. 字=[讲]  出现=12次  平均激活=2.208  上下文示例: 吾讲吾、是讲，、搿讲起、伊讲侬、侬讲是
   6. 字=[葛]  出现=12次  平均激活=1.928  上下文示例: 房葛末、葛末、，葛末
   7. 字=[伊]  出现=12次  平均激活=1.786  上下文示例: 伊还、伊包、呃伊讲、伊拉、伊拿
   8. 字=[侬]  出现=11次  平均激活=2.138  上下文示例: 侬还、光侬要、侬再、侬哪、为侬，
   9. 字=[呢]  出现=11次  平均激活=1.916  上下文示例: 煸呢，、光呢先、拧呢，、司呢，
  10. 字=[开]  出现=11次  平均激活=1.822  上下文示例: 老开开、开开心、开电

### Top 30 Token Evidence (highest activation, per-token detail)
   1. 激活=3.843  字=[呃]  上下文=[呃两]  整句=[呃两点钟啊]
   2. 激活=3.484  字=[是]  上下文=[就是讲]  整句=[嗯，葛末就是讲，嗯，搿呃喃能就每天要吃水果，乃界喃就是讲顶好吃两只]
   3. 激活=3.436  字=[诶]  上下文=[诶，]  整句=[诶，吾还去高伊买点迭呃]
   4. 激活=3.244  字=[讲]  上下文=[吾讲吾]  整句=[吾讲吾对粥特别反感，吾讲吾泡饭倒还没，开水捣捣伊，吾讲吾老要吃]
   5. 激活=3.216  字=[侬]  上下文=[侬还]  整句=[侬还有啥地方去过啊]
   6. 激活=3.210  字=[跌]  上下文=[跌]  整句=[跌]
   7. 激活=3.190  字=[也]  上下文=[也没]  整句=[也没撒去头呃，台湾对伐，嗯，没撒去头，日本呢]
   8. 激活=3.095  字=[以]  上下文=[所以有]  整句=[所以有种经营勒好呃饭店哦，侬看伊呃客座率老是老高呃]
   9. 激活=3.067  字=[末]  上下文=[葛末另]  整句=[葛末另外呢，在加点干点，葛末基本上搿能噶吃]
  10. 激活=3.067  字=[因]  上下文=[伊因为]  整句=[伊因为侬，搿只店里相老是搿两只菜，人家客户就侪跑特辣]
  11. 激活=3.065  字=[出]  上下文=[撩出来]  整句=[撩出来然后呢斩只青椒，青椒呢先煸一煸]
  12. 激活=3.059  字=[低]  上下文=[低价]  整句=[低价位呃股票嗯]
  13. 激活=3.051  字=[呢]  上下文=[煸呢，]  整句=[煸一煸呢，然后侬拿搿洋山芋丝再煸，哎呦搿洋山芋老脆呃老好吃，侬伐相信下趟去试试看]
  14. 激活=3.035  字=[乃]  上下文=[乃末]  整句=[乃末还有嘛调口卡]
  15. 激活=2.971  字=[搿]  上下文=[照搿方]  整句=[还照搿方式方法烧辣，但是伊就讲好像没有小时候呃味道]
  16. 激活=2.960  字=[以]  上下文=[所以有]  整句=[所以有种经营勒好呃饭店哦，侬看伊呃客座率老是老高呃]
  17. 激活=2.942  字=[因]  上下文=[伊因为]  整句=[伊因为侬，搿只店里相老是搿两只菜，人家客户就侪跑特辣]
  18. 激活=2.930  字=[家]  上下文=[人家是]  整句=[人家是讲调料调料，纯在调料]
  19. 激活=2.907  字=[乃]  上下文=[乃末]  整句=[乃末还有嘛调口卡]
  20. 激活=2.864  字=[侬]  上下文=[侬还]  整句=[侬还有啥地方去过啊]
  21. 激活=2.855  字=[末]  上下文=[葛末另]  整句=[葛末另外呢，在加点干点，葛末基本上搿能噶吃]
  22. 激活=2.829  字=[来]  上下文=[出来然]  整句=[撩出来然后呢斩只青椒，青椒呢先煸一煸]
  23. 激活=2.823  字=[吾]  上下文=[，吾吾]  整句=[呃，吾吾晓得撒地方好呃饭店，阿拉也要交流交流，葛末就还是要到好呃地方去]
  24. 激活=2.819  字=[跌]  上下文=[跌到]  整句=[跌到现在诶，就是呢，跌到现在，撒道理伊从五百万，跌到现在只有一百多万了]
  25. 激活=2.810  字=[跌]  上下文=[跌]  整句=[跌]
  26. 激活=2.790  字=[如]  上下文=[如果]  整句=[如果讲搿只饭店拧流量]
  27. 激活=2.778  字=[吾]  上下文=[，吾吾]  整句=[呃，吾吾晓得撒地方好呃饭店，阿拉也要交流交流，葛末就还是要到好呃地方去]
  28. 激活=2.774  字=[葛]  上下文=[房葛末]  整句=[啊里几间包房葛末就拔伊订下来，侬假使讲去了晚呃闲话呢，年夜饭就订伐着了。]
  29. 激活=2.773  字=[讲]  上下文=[吾讲吾]  整句=[吾讲吾对粥特别反感，吾讲吾泡饭倒还没，开水捣捣伊，吾讲吾老要吃]
  30. 激活=2.752  字=[讲]  上下文=[是讲，]  整句=[嗯，葛末就是讲，嗯，搿呃喃能就每天要吃水果，乃界喃就是讲顶好吃两只]

### Acoustic Features Statistics (from top-30 transcripts)
  - snr_db: mean=6.990, std=1.577, n=30
  - speech_rate: mean=18.871, std=10.735, n=30
  - pitch_mean: mean=171.407, std=61.462, n=30
  - pitch_std: mean=41.342, std=27.176, n=30
  - pitch_range: mean=163.919, std=94.484, n=30
  - energy_mean: mean=0.020, std=0.016, n=30
  - energy_std: mean=0.021, std=0.021, n=30
  - energy_dynamic_range: mean=40.479, std=7.488, n=30
  - silence_ratio: mean=0.744, std=0.093, n=30
  - speech_duration: mean=1.046, std=0.589, n=30
  - total_duration: mean=4.025, std=1.919, n=30
  - zero_crossing_rate: mean=0.152, std=0.046, n=30
  - spectral_centroid: mean=1973.944, std=337.851, n=30


### Linguistic Features Statistics (from top-30 transcripts)
  - char_count: mean=20.600, std=11.456
  - word_count: mean=1.000, std=0.000
  - contains_numbers: 0/30 (0.0% True)
  - contains_english: 0/30 (0.0% True)
  - contains_punctuation: 21/30 (70.0% True)
  - number_ratio: mean=0.000, std=0.000
  - contains_person_name: 3/30 (10.0% True)
  - contains_place_name: 0/30 (0.0% True)
  - contains_org_name: 0/30 (0.0% True)
  - voiced_initial_ratio: mean=0.369, std=0.131
  - nasal_final_ratio: mean=0.144, std=0.094


============================================================
REQUIRED OUTPUT FORMAT
============================================================

Return a JSON array with exactly 3 hypotheses:

[
  {
    "Reasoning": "<Detailed explanation referencing specific chars, contexts, and acoustic patterns>",
    "Phenomenon": "<Name of the speech/linguistic phenomenon>",
    "Confidence": "<1-4>"
  },
  ...
]

Confidence Scoring:
- 4: Character distribution + context patterns + acoustic trends all strongly align
- 3: Most evidence aligns, minor inconsistencies
- 2: Some evidence aligns, weaker support
- 1: Very little aligns, highly uncertain

============================================================
CHECKLIST FOR REASONING (address each item)
============================================================

1. **Character Pattern Analysis**:
   - Which characters dominate the distribution? Are they phonetically similar?
   - Do the context windows (aligned_context) reveal consistent phonetic neighbors?
   - Are there repeated syllable structures or tonal patterns across contexts?

2. **Context Window Interpretation**:
   - What characters appear before/after the aligned character most often?
   - Do context patterns suggest word boundaries, tone sandhi, or co-articulation effects?

3. **Acoustic Feature Interpretation**:
   - Do energy/pitch patterns suggest specific phonetic categories (voiced/unvoiced, tonal)?
   - Does spectral centroid or zero_crossing_rate indicate fricatives, stops, or nasals?

4. **Dialect-Specific Analysis**:
   - Are the high-frequency characters Shanghainese-specific vocabulary?
   - Do context patterns suggest Wu dialect tone sandhi rules?

5. **Linguistic Level**:
   - Is this a phoneme-level, syllable-level, word-level, or sentence-level feature?

6. **Error Pattern Correlation**:
   - Could this neuron's activation pattern relate to specific ASR confusion types?
   - Homophone errors? Tone errors? Word boundary errors?

======================================================================


======================================================================
You are given data from a Whisper-based Automatic Speech Recognition (ASR) model fine-tuned for Shanghainese (上海话) dialect.

Model Logic:
- Input: Raw audio waveform (Shanghainese speech)
- Encoder: Extracts acoustic features → hidden representations
- SAE Layer: Sparse Autoencoder applied to encoder layer 12 hidden states
- Decoder: Generates text transcription from encoded features
- Output: Chinese text transcription

The provided data describes a **concept** extracted by a Sparse Autoencoder (SAE).
A concept = a speech/linguistic feature detected from the audio that influences transcription output.

Your Task:
Using the character distribution, context windows, and acoustic/linguistic statistics below,
identify **three possible speech/linguistic phenomena** this neuron concept could represent,
ranked in descending order of confidence.

============================================================
DATA FORMAT DEFINITIONS
============================================================

1. **Character Distribution** (主线):
   The most frequent characters (aligned_char) among the top-500 highest-activation tokens.
   Each entry shows: character, occurrence count, mean activation value, and context window examples.
   The context window (aligned_context) is the ±1 character window around the aligned character
   in the transcript, giving local phonetic/lexical context.

2. **Top 30 Token Evidence** (具体证据):
   The 30 individual tokens with highest activation values.
   Each entry shows: activation value, the aligned character, its context window, and the full transcript.
   This preserves per-token detail so you can spot patterns not visible in aggregated statistics.

3. **Acoustic Features**: Audio-level statistics (energy, pitch, spectral centroid, etc.)
   computed over the full transcript of each top-activation token.

4. **Linguistic Features**: Text-level statistics (char count, punctuation, dialect markers, etc.)
   computed over the full transcript of each top-activation token.

============================================================
CURRENT CONCEPT DATA: Neuron #1003
(Total activation events: 3989)
============================================================

### Character Distribution (Top 10 chars, from top-500 tokens)
   1. 字=[，]  出现=40次  平均激活=3.071  上下文示例: 诶，大、做，伊、师，吾、搭，就、补，侬
   2. 字=[呃]  出现=18次  平均激活=3.005  上下文示例: 是呃，、烧呃搿、性呃撒、侬呃材、呃两
   3. 字=[葛]  出现=16次  平均激活=3.368  上下文示例: 葛末、，葛末
   4. 字=[是]  出现=14次  平均激活=4.034  上下文示例: 就是服、就是该、讲是呃、就是呢、侪是那
   5. 字=[伐]  出现=13次  平均激活=3.484  上下文示例: 末伐然、用伐着、对伐也、对伐啦、对伐要
   6. 字=[就]  出现=12次  平均激活=3.381  上下文示例: 啦就讲、搿就对、拉就没、，就就、就是
   7. 字=[搿]  出现=11次  平均激活=3.429  上下文示例: 呃搿搿、但搿朋、是搿能、搿房、搿呃
   8. 字=[想]  出现=10次  平均激活=3.330  上下文示例: 要想订、吾想想、要想办
   9. 字=[伊]  出现=9次  平均激活=3.488  上下文示例: 伊讲、伊拉、伊总、伊伐、，伊也
  10. 字=[讲]  出现=9次  平均激活=3.302  上下文示例: 该讲是、就讲搿、要讲唻、伊讲阿

### Top 30 Token Evidence (highest activation, per-token detail)
   1. 激活=8.140  字=[现]  上下文=[就现在]  整句=[现就现在呐就讲好像侪到了搿呃年龄了，现在撒跑伐动了]
   2. 激活=7.963  字=[日]  上下文=[半日就]  整句=[进去，或者下半日就拿出来，也有呃，或者明朝拿出来，也有呃，伊搿只产品只有只利率低]
   3. 激活=7.820  字=[也]  上下文=[也是]  整句=[也是搿呃价钿咯]
   4. 激活=7.615  字=[以]  上下文=[所以有]  整句=[所以有辰光要叫阿拉带伊拉带小拧咯撒，伊小拧的话，第三代呃小拧呐也蛮难带呃]
   5. 激活=7.411  字=[以]  上下文=[所以有]  整句=[所以有辰光要叫阿拉带伊拉带小拧咯撒，伊小拧的话，第三代呃小拧呐也蛮难带呃]
   6. 激活=7.020  字=[是]  上下文=[就是服]  整句=[搿就是服务呃，星级服务呃搿呃，标准上去了呀。]
   7. 激活=6.894  字=[空]  上下文=[末空辣]  整句=[葛末空辣海么就就没收益嘞，伊拉就会得想出了搿办法]
   8. 激活=6.659  字=[一]  上下文=[某一呃]  整句=[某一呃拧做事体，侪是辣埃呃，侪失败中教训能成长]
   9. 激活=6.637  字=[是]  上下文=[就是该]  整句=[彻底蹲辣埃面搭，就就是该哪能孛相，就哪能孛相该哪能吃就哪能吃哪能来就哪能来搿是真呃是老好一庄事体]
  10. 激活=6.606  字=[葛]  上下文=[葛末]  整句=[葛末对阿拉对阿拉来讲喃既做到]
  11. 激活=6.547  字=[伐]  上下文=[末伐然]  整句=[葛末伐然假使讲搿场地就空关勒海，葛末吃呃拧老少。呃]
  12. 激活=6.546  字=[拧]  上下文=[拧也]  整句=[拧也伐来唻，葛末老两口呢就就自家吃吃饭，觉着呢就讲好像生活呢就讲忒厌气了晓得伐]
  13. 激活=6.520  字=[喃]  上下文=[饭喃，]  整句=[订年夜饭喃，一般伊来了中秋节前后就开始要订了啦，订了以后]
  14. 激活=6.510  字=[，]  上下文=[诶，大]  整句=[诶，大家全新合力呃辣该做，呃各个分工辣该做，做了还是]
  15. 激活=6.473  字=[诶]  上下文=[诶，]  整句=[诶，搿辰光吾辣辣跟伊辣埃面做呀]
  16. 激活=6.438  字=[趟]  上下文=[下趟下]  整句=[下趟下一代伊拉就没搿种叫法了，晓得伐，所以阿拉呢现在搿一代呐就讲好像勒最后一代]
  17. 激活=6.224  字=[是]  上下文=[讲是呃]  整句=[应该讲是呃伐大会的埃呃，因为伊拉开到现在几十年唻]
  18. 激活=6.195  字=[伊]  上下文=[伊讲]  整句=[伊讲开了三四年]
  19. 激活=6.121  字=[子]  上下文=[年子侪]  整句=[哎今年子侪要伐断呃来，来改变自家呃搿种经营策略哦]
  20. 激活=6.031  字=[客]  上下文=[对客户]  整句=[对客户来讲喃，伊穿勒阿拉漂亮呃衣裳喃，也是无形当中为阿拉企业喃做广告]
  21. 激活=6.022  字=[搿]  上下文=[呃搿搿]  整句=[呃搿搿能噶了，所以现在年纪老呃拧呐像阿拉呣妈搿搭代呢就讲还有的介许多拧好埃呃叫撒，呃现在]
  22. 激活=5.976  字=[个]  上下文=[整个上]  整句=[整个上海伐是侪好走了嘛，阿拉是，喏，喏，现现在讲起来莘庄]
  23. 激活=5.962  字=[对]  上下文=[对伐]  整句=[对伐，迭呃老年拧没迭呃迭呃叫搿能一呃绝对权威了]
  24. 激活=5.950  字=[记]  上下文=[搿记叫]  整句=[搿记叫啥，烧辣伐好对伐，伊拉讲，吾就觉得烧辣伐好，对伐]
  25. 激活=5.910  字=[因]  上下文=[因为]  整句=[因为吾看伊拉几呃阿姨辣辣买呃辰光]
  26. 激活=5.834  字=[是]  上下文=[就是呢]  整句=[跌到现在诶，就是呢，跌到现在，撒道理伊从五百万，跌到现在只有一百多万了]
  27. 激活=5.787  字=[跑]  上下文=[末跑到]  整句=[乃末跑到外头去喃，阿拉代表阿拉阿爸爸搿一房呢，阿拉屋里相呃兄弟姐妹阿拉爸爸呃子女呢侪要团结好]
  28. 激活=5.621  字=[不]  上下文=[不伦]  整句=[不伦不类去养了呃蛇也有]
  29. 激活=5.606  字=[记]  上下文=[搿记叫]  整句=[搿记叫啥，烧辣伐好对伐，伊拉讲，吾就觉得烧辣伐好，对伐]
  30. 激活=5.569  字=[，]  上下文=[做，伊]  整句=[做，伊是医院里做呃，伊工作老吃力呃]

### Acoustic Features Statistics (from top-30 transcripts)
  - snr_db: mean=6.008, std=1.022, n=30
  - speech_rate: mean=16.594, std=3.626, n=30
  - pitch_mean: mean=181.028, std=41.324, n=30
  - pitch_std: mean=43.143, std=24.096, n=30
  - pitch_range: mean=200.342, std=121.811, n=30
  - energy_mean: mean=0.041, std=0.027, n=30
  - energy_std: mean=0.038, std=0.027, n=30
  - energy_dynamic_range: mean=42.184, std=5.993, n=30
  - silence_ratio: mean=0.684, std=0.062, n=30
  - speech_duration: mean=1.550, std=0.599, n=30
  - total_duration: mean=4.950, std=1.717, n=30
  - zero_crossing_rate: mean=0.147, std=0.036, n=30
  - spectral_centroid: mean=1935.834, std=323.381, n=30


### Linguistic Features Statistics (from top-30 transcripts)
  - char_count: mean=27.000, std=10.814
  - word_count: mean=1.000, std=0.000
  - contains_numbers: 0/30 (0.0% True)
  - contains_english: 0/30 (0.0% True)
  - contains_punctuation: 25/30 (83.3% True)
  - number_ratio: mean=0.000, std=0.000
  - contains_person_name: 0/30 (0.0% True)
  - contains_place_name: 1/30 (3.3% True)
  - contains_org_name: 1/30 (3.3% True)
  - voiced_initial_ratio: mean=0.447, std=0.079
  - nasal_final_ratio: mean=0.122, std=0.070


============================================================
REQUIRED OUTPUT FORMAT
============================================================

Return a JSON array with exactly 3 hypotheses:

[
  {
    "Reasoning": "<Detailed explanation referencing specific chars, contexts, and acoustic patterns>",
    "Phenomenon": "<Name of the speech/linguistic phenomenon>",
    "Confidence": "<1-4>"
  },
  ...
]

Confidence Scoring:
- 4: Character distribution + context patterns + acoustic trends all strongly align
- 3: Most evidence aligns, minor inconsistencies
- 2: Some evidence aligns, weaker support
- 1: Very little aligns, highly uncertain

============================================================
CHECKLIST FOR REASONING (address each item)
============================================================

1. **Character Pattern Analysis**:
   - Which characters dominate the distribution? Are they phonetically similar?
   - Do the context windows (aligned_context) reveal consistent phonetic neighbors?
   - Are there repeated syllable structures or tonal patterns across contexts?

2. **Context Window Interpretation**:
   - What characters appear before/after the aligned character most often?
   - Do context patterns suggest word boundaries, tone sandhi, or co-articulation effects?

3. **Acoustic Feature Interpretation**:
   - Do energy/pitch patterns suggest specific phonetic categories (voiced/unvoiced, tonal)?
   - Does spectral centroid or zero_crossing_rate indicate fricatives, stops, or nasals?

4. **Dialect-Specific Analysis**:
   - Are the high-frequency characters Shanghainese-specific vocabulary?
   - Do context patterns suggest Wu dialect tone sandhi rules?

5. **Linguistic Level**:
   - Is this a phoneme-level, syllable-level, word-level, or sentence-level feature?

6. **Error Pattern Correlation**:
   - Could this neuron's activation pattern relate to specific ASR confusion types?
   - Homophone errors? Tone errors? Word boundary errors?

======================================================================


======================================================================
You are given data from a Whisper-based Automatic Speech Recognition (ASR) model fine-tuned for Shanghainese (上海话) dialect.

Model Logic:
- Input: Raw audio waveform (Shanghainese speech)
- Encoder: Extracts acoustic features → hidden representations
- SAE Layer: Sparse Autoencoder applied to encoder layer 12 hidden states
- Decoder: Generates text transcription from encoded features
- Output: Chinese text transcription

The provided data describes a **concept** extracted by a Sparse Autoencoder (SAE).
A concept = a speech/linguistic feature detected from the audio that influences transcription output.

Your Task:
Using the character distribution, context windows, and acoustic/linguistic statistics below,
identify **three possible speech/linguistic phenomena** this neuron concept could represent,
ranked in descending order of confidence.

============================================================
DATA FORMAT DEFINITIONS
============================================================

1. **Character Distribution** (主线):
   The most frequent characters (aligned_char) among the top-500 highest-activation tokens.
   Each entry shows: character, occurrence count, mean activation value, and context window examples.
   The context window (aligned_context) is the ±1 character window around the aligned character
   in the transcript, giving local phonetic/lexical context.

2. **Top 30 Token Evidence** (具体证据):
   The 30 individual tokens with highest activation values.
   Each entry shows: activation value, the aligned character, its context window, and the full transcript.
   This preserves per-token detail so you can spot patterns not visible in aggregated statistics.

3. **Acoustic Features**: Audio-level statistics (energy, pitch, spectral centroid, etc.)
   computed over the full transcript of each top-activation token.

4. **Linguistic Features**: Text-level statistics (char count, punctuation, dialect markers, etc.)
   computed over the full transcript of each top-activation token.

============================================================
CURRENT CONCEPT DATA: Neuron #650
(Total activation events: 5848)
============================================================

### Character Distribution (Top 10 chars, from top-500 tokens)
   1. 字=[只]  出现=18次  平均激活=2.296  上下文示例: 伐只有
   2. 字=[有]  出现=16次  平均激活=2.152  上下文示例: 还有嘛、有呃、然有场、拧有常、呃有呃
   3. 字=[，]  出现=15次  平均激活=2.093  上下文示例: 价，诶、呢，大、呃，基、诶，脑、哎，搿
   4. 字=[呃]  出现=15次  平均激活=2.011  上下文示例: 迭呃里、四呃五、头呃情、有呃有、呃，
   5. 字=[是]  出现=15次  平均激活=2.053  上下文示例: 但是伊、嘛是发、为是搿、是有、妇是侬
   6. 字=[侬]  出现=13次  平均激活=2.182  上下文示例: 侬，、侬好、侬搿、为侬，
   7. 字=[嗯]  出现=12次  平均激活=2.096  上下文示例: 嗯，、嗯阿
   8. 字=[烧]  出现=11次  平均激活=2.172  上下文示例: 辣烧呃、烧毛
   9. 字=[末]  出现=11次  平均激活=2.145  上下文示例: 葛末空、乃末养、乃末侬、葛末，、乃末挨
  10. 字=[诶]  出现=10次  平均激活=2.151  上下文示例: 诶，、诶诶、，诶些

### Top 30 Token Evidence (highest activation, per-token detail)
   1. 激活=3.019  字=[毛]  上下文=[烧毛豆]  整句=[烧毛豆前头，侬伐剪搿毛豆侪剪好伐]
   2. 激活=2.966  字=[除]  上下文=[除除除]  整句=[葛末喏除除除除特买点鸡啊，买点鸭啊，一只鸽子蛮好呃]
   3. 激活=2.961  字=[只]  上下文=[伐只有]  整句=[吾对伐只有早哴向老早爬起来，拿呃洋山芋，嗒嗒嗒嗒嗒嗒嗒]
   4. 激活=2.914  字=[像]  上下文=[拉像，]  整句=[对呃呀，阿拉像，阿拉，吾呃爸爸妈妈侪走特了，走特了嘛，阿拉爸爸现在现在要一百岁唻]
   5. 激活=2.881  字=[约]  上下文=[节约伊]  整句=[诶，真呃老节约伊讲，伊拉埃面外头撒呃奶茶咯，撒呃，诶，撒呃，诶种噶撒呃]
   6. 激活=2.856  字=[只]  上下文=[伐只有]  整句=[吾对伐只有早哴向老早爬起来，拿呃洋山芋，嗒嗒嗒嗒嗒嗒嗒]
   7. 激活=2.829  字=[能]  上下文=[可能是]  整句=[晓晓得有可能是伐会的好，但是想伐大可能老板跑特伐，因为当初辰光跑特呃老板必定还老少呃]
   8. 激活=2.811  字=[烧]  上下文=[辣烧呃]  整句=[辣烧呃前头侬先冷罐冷开水辣辣旁边]
   9. 激活=2.787  字=[人]  上下文=[吃人家]  整句=[哦吃人家糊糊，现在讲起来是小米，现在讲起来好唻]
  10. 激活=2.741  字=[侬]  上下文=[侬，]  整句=[侬，吾老早一直是烧呃，吾是讲荤荤呃汤吾伐大考虑呃]
  11. 激活=2.729  字=[老]  上下文=[吾老早]  整句=[侬，吾老早一直是烧呃，吾是讲荤荤呃汤吾伐大考虑呃]
  12. 激活=2.728  字=[有]  上下文=[还有嘛]  整句=[想还有嘛，伊想到伊呃爷娘，还伐会想到侬老头老太]
  13. 激活=2.713  字=[只]  上下文=[伐只有]  整句=[吾对伐只有早哴向老早爬起来，拿呃洋山芋，嗒嗒嗒嗒嗒嗒嗒]
  14. 激活=2.674  字=[末]  上下文=[葛末空]  整句=[葛末空辣海么就就没收益嘞，伊拉就会得想出了搿办法]
  15. 激活=2.659  字=[诶]  上下文=[诶，]  整句=[诶，搿辰光吾辣辣跟伊辣埃面做呀]
  16. 激活=2.649  字=[伊]  上下文=[伊会]  整句=[伊会的来问那呃，那去伐]
  17. 激活=2.641  字=[烧]  上下文=[辣烧呃]  整句=[辣烧呃前头侬先冷罐冷开水辣辣旁边]
  18. 激活=2.634  字=[切]  上下文=[切下]  整句=[切下来然后对伐，就摆眼姜摆眼蛋清，嗯]
  19. 激活=2.630  字=[讲]  上下文=[要讲唻]  整句=[葛末吾吾要讲唻，葛末侬股票哪能介埃呃啊，股票觉着哪能介好呃啦]
  20. 激活=2.621  字=[只]  上下文=[伐只有]  整句=[吾对伐只有早哴向老早爬起来，拿呃洋山芋，嗒嗒嗒嗒嗒嗒嗒]
  21. 激活=2.620  字=[只]  上下文=[伐只有]  整句=[吾对伐只有早哴向老早爬起来，拿呃洋山芋，嗒嗒嗒嗒嗒嗒嗒]
  22. 激活=2.604  字=[伊]  上下文=[伊会]  整句=[伊会的来问那呃，那去伐]
  23. 激活=2.594  字=[为]  上下文=[因为产]  整句=[因为产品伐合格，侬就作废特了报废特了]
  24. 激活=2.580  字=[约]  上下文=[节约伊]  整句=[诶，真呃老节约伊讲，伊拉埃面外头撒呃奶茶咯，撒呃，诶，撒呃，诶种噶撒呃]
  25. 激活=2.570  字=[侬]  上下文=[侬好]  整句=[侬好买，为撒道理呢]
  26. 激活=2.566  字=[到]  上下文=[跌到现]  整句=[跌到现在诶，就是呢，跌到现在，撒道理伊从五百万，跌到现在只有一百多万了]
  27. 激活=2.551  字=[种]  上下文=[有种专]  整句=[有种专门吃吃饲料呃，吾怀疑，饲料上头有点有点区别]
  28. 激活=2.546  字=[哦]  上下文=[哦今]  整句=[哦今朝阿拉讲勒蛮开心呃，从吃讲到穿讲到孛相]
  29. 激活=2.534  字=[啊]  上下文=[啊阿]  整句=[啊阿拉是上趟子有一年是啊里一年啊]
  30. 激活=2.528  字=[烧]  上下文=[辣烧呃]  整句=[辣烧呃前头侬先冷罐冷开水辣辣旁边]

### Acoustic Features Statistics (from top-30 transcripts)
  - snr_db: mean=5.893, std=0.891, n=30
  - speech_rate: mean=22.330, std=6.924, n=30
  - pitch_mean: mean=175.570, std=41.783, n=30
  - pitch_std: mean=37.800, std=22.337, n=30
  - pitch_range: mean=166.203, std=100.225, n=30
  - energy_mean: mean=0.021, std=0.020, n=30
  - energy_std: mean=0.020, std=0.021, n=30
  - energy_dynamic_range: mean=35.187, std=8.233, n=30
  - silence_ratio: mean=0.758, std=0.075, n=30
  - speech_duration: mean=1.082, std=0.561, n=30
  - total_duration: mean=4.381, std=1.466, n=30
  - zero_crossing_rate: mean=0.164, std=0.046, n=30
  - spectral_centroid: mean=2000.339, std=354.493, n=30


### Linguistic Features Statistics (from top-30 transcripts)
  - char_count: mean=23.400, std=8.289
  - word_count: mean=1.000, std=0.000
  - contains_numbers: 0/30 (0.0% True)
  - contains_english: 0/30 (0.0% True)
  - contains_punctuation: 26/30 (86.7% True)
  - number_ratio: mean=0.000, std=0.000
  - contains_person_name: 0/30 (0.0% True)
  - contains_place_name: 1/30 (3.3% True)
  - contains_org_name: 0/30 (0.0% True)
  - voiced_initial_ratio: mean=0.434, std=0.131
  - nasal_final_ratio: mean=0.149, std=0.077


============================================================
REQUIRED OUTPUT FORMAT
============================================================

Return a JSON array with exactly 3 hypotheses:

[
  {
    "Reasoning": "<Detailed explanation referencing specific chars, contexts, and acoustic patterns>",
    "Phenomenon": "<Name of the speech/linguistic phenomenon>",
    "Confidence": "<1-4>"
  },
  ...
]

Confidence Scoring:
- 4: Character distribution + context patterns + acoustic trends all strongly align
- 3: Most evidence aligns, minor inconsistencies
- 2: Some evidence aligns, weaker support
- 1: Very little aligns, highly uncertain

============================================================
CHECKLIST FOR REASONING (address each item)
============================================================

1. **Character Pattern Analysis**:
   - Which characters dominate the distribution? Are they phonetically similar?
   - Do the context windows (aligned_context) reveal consistent phonetic neighbors?
   - Are there repeated syllable structures or tonal patterns across contexts?

2. **Context Window Interpretation**:
   - What characters appear before/after the aligned character most often?
   - Do context patterns suggest word boundaries, tone sandhi, or co-articulation effects?

3. **Acoustic Feature Interpretation**:
   - Do energy/pitch patterns suggest specific phonetic categories (voiced/unvoiced, tonal)?
   - Does spectral centroid or zero_crossing_rate indicate fricatives, stops, or nasals?

4. **Dialect-Specific Analysis**:
   - Are the high-frequency characters Shanghainese-specific vocabulary?
   - Do context patterns suggest Wu dialect tone sandhi rules?

5. **Linguistic Level**:
   - Is this a phoneme-level, syllable-level, word-level, or sentence-level feature?

6. **Error Pattern Correlation**:
   - Could this neuron's activation pattern relate to specific ASR confusion types?
   - Homophone errors? Tone errors? Word boundary errors?

======================================================================


======================================================================
You are given data from a Whisper-based Automatic Speech Recognition (ASR) model fine-tuned for Shanghainese (上海话) dialect.

Model Logic:
- Input: Raw audio waveform (Shanghainese speech)
- Encoder: Extracts acoustic features → hidden representations
- SAE Layer: Sparse Autoencoder applied to encoder layer 12 hidden states
- Decoder: Generates text transcription from encoded features
- Output: Chinese text transcription

The provided data describes a **concept** extracted by a Sparse Autoencoder (SAE).
A concept = a speech/linguistic feature detected from the audio that influences transcription output.

Your Task:
Using the character distribution, context windows, and acoustic/linguistic statistics below,
identify **three possible speech/linguistic phenomena** this neuron concept could represent,
ranked in descending order of confidence.

============================================================
DATA FORMAT DEFINITIONS
============================================================

1. **Character Distribution** (主线):
   The most frequent characters (aligned_char) among the top-500 highest-activation tokens.
   Each entry shows: character, occurrence count, mean activation value, and context window examples.
   The context window (aligned_context) is the ±1 character window around the aligned character
   in the transcript, giving local phonetic/lexical context.

2. **Top 30 Token Evidence** (具体证据):
   The 30 individual tokens with highest activation values.
   Each entry shows: activation value, the aligned character, its context window, and the full transcript.
   This preserves per-token detail so you can spot patterns not visible in aggregated statistics.

3. **Acoustic Features**: Audio-level statistics (energy, pitch, spectral centroid, etc.)
   computed over the full transcript of each top-activation token.

4. **Linguistic Features**: Text-level statistics (char count, punctuation, dialect markers, etc.)
   computed over the full transcript of each top-activation token.

============================================================
CURRENT CONCEPT DATA: Neuron #4990
(Total activation events: 18)
============================================================

### Character Distribution (Top 10 chars, from top-500 tokens)
   1. 字=[票]  出现=3次  平均激活=1.302  上下文示例: 钞票
   2. 字=[放]  出现=3次  平均激活=1.216  上下文示例: 放侬
   3. 字=[候]  出现=2次  平均激活=1.314  上下文示例: 时候了
   4. 字=[时]  出现=2次  平均激活=1.333  上下文示例: 的时候
   5. 字=[嘛]  出现=1次  平均激活=1.301  上下文示例: 鱼嘛，
   6. 字=[往]  出现=1次  平均激活=1.286  上下文示例: 往往
   7. 字=[明]  出现=1次  平均激活=1.285  上下文示例: 明天
   8. 字=[上]  出现=1次  平均激活=1.249  上下文示例: 加上我
   9. 字=[年]  出现=1次  平均激活=1.206  上下文示例: 四年
  10. 字=[行]  出现=1次  平均激活=1.177  上下文示例: 银行里

### Top 30 Token Evidence (highest activation, per-token detail)
   1. 激活=1.432  字=[候]  上下文=[时候了]  整句=[明天吃鱼嘛，后天又想吃肉，再加上我们这种年龄啊，好像是养老的时候了]
   2. 激活=1.411  字=[票]  上下文=[钞票]  整句=[放侬银行里钞票]
   3. 激活=1.337  字=[时]  上下文=[的时候]  整句=[明天吃鱼嘛，后天又想吃肉，再加上我们这种年龄啊，好像是养老的时候了]
   4. 激活=1.328  字=[时]  上下文=[的时候]  整句=[明天吃鱼嘛，后天又想吃肉，再加上我们这种年龄啊，好像是养老的时候了]
   5. 激活=1.301  字=[嘛]  上下文=[鱼嘛，]  整句=[明天吃鱼嘛，后天又想吃肉，再加上我们这种年龄啊，好像是养老的时候了]
   6. 激活=1.286  字=[往]  上下文=[往往]  整句=[现在呢往往]
   7. 激活=1.285  字=[票]  上下文=[钞票]  整句=[放侬银行里钞票]
   8. 激活=1.285  字=[明]  上下文=[明天]  整句=[明天吃鱼嘛，后天又想吃肉，再加上我们这种年龄啊，好像是养老的时候了]
   9. 激活=1.250  字=[放]  上下文=[放侬]  整句=[放侬银行里钞票]
  10. 激活=1.249  字=[上]  上下文=[加上我]  整句=[明天吃鱼嘛，后天又想吃肉，再加上我们这种年龄啊，好像是养老的时候了]
  11. 激活=1.244  字=[放]  上下文=[放侬]  整句=[放侬银行里钞票]
  12. 激活=1.211  字=[票]  上下文=[钞票]  整句=[放侬银行里钞票]
  13. 激活=1.206  字=[年]  上下文=[四年]  整句=[伊讲开了三四年]
  14. 激活=1.197  字=[候]  上下文=[时候了]  整句=[明天吃鱼嘛，后天又想吃肉，再加上我们这种年龄啊，好像是养老的时候了]
  15. 激活=1.177  字=[行]  上下文=[银行里]  整句=[放侬银行里钞票]
  16. 激活=1.159  字=[高]  上下文=[吾高阿]  整句=[吾高阿拉屋里相讲，侬伐一天到晚搿种鸡汤鸭汤，伊昨日还是搞呃鸡汤]
  17. 激活=1.154  字=[放]  上下文=[放侬]  整句=[放侬银行里钞票]
  18. 激活=1.077  字=[里]  上下文=[行里钞]  整句=[放侬银行里钞票]

### Acoustic Features Statistics (from top-30 transcripts)
  - snr_db: mean=7.222, std=0.905, n=18
  - speech_rate: mean=15.039, std=3.577, n=18
  - pitch_mean: mean=168.840, std=18.531, n=18
  - pitch_std: mean=40.486, std=21.479, n=18
  - pitch_range: mean=136.665, std=77.466, n=18
  - energy_mean: mean=0.033, std=0.013, n=18
  - energy_std: mean=0.031, std=0.011, n=18
  - energy_dynamic_range: mean=41.143, std=5.362, n=18
  - silence_ratio: mean=0.638, std=0.121, n=18
  - speech_duration: mean=1.039, std=0.538, n=18
  - total_duration: mean=3.729, std=2.868, n=18
  - zero_crossing_rate: mean=0.134, std=0.037, n=18
  - spectral_centroid: mean=1815.581, std=462.719, n=18


### Linguistic Features Statistics (from top-30 transcripts)
  - char_count: mean=18.333, std=12.910
  - word_count: mean=1.000, std=0.000
  - contains_numbers: 0/18 (0.0% True)
  - contains_english: 0/18 (0.0% True)
  - contains_punctuation: 8/18 (44.4% True)
  - number_ratio: mean=0.000, std=0.000
  - contains_person_name: 1/18 (5.6% True)
  - contains_place_name: 0/18 (0.0% True)
  - contains_org_name: 8/18 (44.4% True)
  - voiced_initial_ratio: mean=0.522, std=0.076
  - nasal_final_ratio: mean=0.401, std=0.158


============================================================
REQUIRED OUTPUT FORMAT
============================================================

Return a JSON array with exactly 3 hypotheses:

[
  {
    "Reasoning": "<Detailed explanation referencing specific chars, contexts, and acoustic patterns>",
    "Phenomenon": "<Name of the speech/linguistic phenomenon>",
    "Confidence": "<1-4>"
  },
  ...
]

Confidence Scoring:
- 4: Character distribution + context patterns + acoustic trends all strongly align
- 3: Most evidence aligns, minor inconsistencies
- 2: Some evidence aligns, weaker support
- 1: Very little aligns, highly uncertain

============================================================
CHECKLIST FOR REASONING (address each item)
============================================================

1. **Character Pattern Analysis**:
   - Which characters dominate the distribution? Are they phonetically similar?
   - Do the context windows (aligned_context) reveal consistent phonetic neighbors?
   - Are there repeated syllable structures or tonal patterns across contexts?

2. **Context Window Interpretation**:
   - What characters appear before/after the aligned character most often?
   - Do context patterns suggest word boundaries, tone sandhi, or co-articulation effects?

3. **Acoustic Feature Interpretation**:
   - Do energy/pitch patterns suggest specific phonetic categories (voiced/unvoiced, tonal)?
   - Does spectral centroid or zero_crossing_rate indicate fricatives, stops, or nasals?

4. **Dialect-Specific Analysis**:
   - Are the high-frequency characters Shanghainese-specific vocabulary?
   - Do context patterns suggest Wu dialect tone sandhi rules?

5. **Linguistic Level**:
   - Is this a phoneme-level, syllable-level, word-level, or sentence-level feature?

6. **Error Pattern Correlation**:
   - Could this neuron's activation pattern relate to specific ASR confusion types?
   - Homophone errors? Tone errors? Word boundary errors?

======================================================================


======================================================================
You are given data from a Whisper-based Automatic Speech Recognition (ASR) model fine-tuned for Shanghainese (上海话) dialect.

Model Logic:
- Input: Raw audio waveform (Shanghainese speech)
- Encoder: Extracts acoustic features → hidden representations
- SAE Layer: Sparse Autoencoder applied to encoder layer 12 hidden states
- Decoder: Generates text transcription from encoded features
- Output: Chinese text transcription

The provided data describes a **concept** extracted by a Sparse Autoencoder (SAE).
A concept = a speech/linguistic feature detected from the audio that influences transcription output.

Your Task:
Using the character distribution, context windows, and acoustic/linguistic statistics below,
identify **three possible speech/linguistic phenomena** this neuron concept could represent,
ranked in descending order of confidence.

============================================================
DATA FORMAT DEFINITIONS
============================================================

1. **Character Distribution** (主线):
   The most frequent characters (aligned_char) among the top-500 highest-activation tokens.
   Each entry shows: character, occurrence count, mean activation value, and context window examples.
   The context window (aligned_context) is the ±1 character window around the aligned character
   in the transcript, giving local phonetic/lexical context.

2. **Top 30 Token Evidence** (具体证据):
   The 30 individual tokens with highest activation values.
   Each entry shows: activation value, the aligned character, its context window, and the full transcript.
   This preserves per-token detail so you can spot patterns not visible in aggregated statistics.

3. **Acoustic Features**: Audio-level statistics (energy, pitch, spectral centroid, etc.)
   computed over the full transcript of each top-activation token.

4. **Linguistic Features**: Text-level statistics (char count, punctuation, dialect markers, etc.)
   computed over the full transcript of each top-activation token.

============================================================
CURRENT CONCEPT DATA: Neuron #2979
(Total activation events: 7138)
============================================================

### Character Distribution (Top 10 chars, from top-500 tokens)
   1. 字=[伊]  出现=30次  平均激活=4.634  上下文示例: 伊拉、伊讲、伊屋、伊伐、伊因
   2. 字=[，]  出现=18次  平均激活=4.554  上下文示例: 末，吾、慰，为、伊，一、婚，乃、诶，吾
   3. 字=[呃]  出现=14次  平均激活=4.810  上下文示例: 三呃，、侬呃材、呃两、性呃撒、一呃要
   4. 字=[侬]  出现=14次  平均激活=4.521  上下文示例: 侬刚、光侬要、侬伐、叫侬签、末侬做
   5. 字=[吾]  出现=13次  平均激活=4.634  上下文示例: 没吾是、为吾做、吾就、吾现、，吾买
   6. 字=[末]  出现=13次  平均激活=4.914  上下文示例: 乃末挨、乃末侬、葛末另、葛末吾、乃末另
   7. 字=[有]  出现=11次  平均激活=4.752  上下文示例: 有种、有点、有辰、还有交、以有种
   8. 字=[葛]  出现=10次  平均激活=4.964  上下文示例: 葛末
   9. 字=[搿]  出现=10次  平均激活=4.513  上下文示例: ，搿能、搿叫、侬搿，、拿搿搿、搿搿就
  10. 字=[就]  出现=10次  平均激活=4.764  上下文示例: 就算、啦就讲、就是、就讲

### Top 30 Token Evidence (highest activation, per-token detail)
   1. 激活=7.309  字=[好]  上下文=[择好侬]  整句=[选择好侬好看得到伊搿排排名啊，伊搿呃就是讲]
   2. 激活=6.964  字=[讲]  上下文=[伊讲侬]  整句=[搿房子老好呃伊讲侬放心好了，那放心好了，现在呢吾伐那做长期呃，做短期呃，]
   3. 激活=6.885  字=[有]  上下文=[有种]  整句=[有种地方伐好看]
   4. 激活=6.820  字=[有]  上下文=[有点]  整句=[有点伐大好了，但是伊当然伐会的讲伐好呃，讲了伐好了，那要人心惶惶伐，伊就还讲自家公司老好呃]
   5. 激活=6.577  字=[起]  上下文=[讲起来]  整句=[对伐啦。现在讲起来伐稀奇唻，现在册那假使要调查侬老便当呃，一只手机，一只电脑马上就甩过去了]
   6. 激活=6.519  字=[正]  上下文=[蛮正确]  整句=[吾觉着是也蛮正确呃种方案，但是吾上趟走进呃一家公司，伊是搞国际物流呃]
   7. 激活=6.472  字=[为]  上下文=[因为吾]  整句=[因为吾看伊拉几呃阿姨辣辣买呃辰光]
   8. 激活=6.431  字=[去]  上下文=[娘去，]  整句=[侪是那娘去，两三线城市或者撒，大城市呃郊区呃地方[+]]
   9. 激活=6.424  字=[择]  上下文=[选择好]  整句=[选择好侬好看得到伊搿排排名啊，伊搿呃就是讲]
  10. 激活=6.400  字=[去]  上下文=[娘去，]  整句=[侪是那娘去，两三线城市或者撒，大城市呃郊区呃地方[+]]
  11. 激活=6.381  字=[择]  上下文=[选择好]  整句=[选择好侬好看得到伊搿排排名啊，伊搿呃就是讲]
  12. 激活=6.344  字=[啦]  上下文=[婆啦就]  整句=[会的痛老婆啦就讲。所以人家讲起来上海呃男呃是，因为人家现在寻，拧家老早讲啦]
  13. 激活=6.242  字=[煸]  上下文=[煸一]  整句=[煸一煸呢，然后侬拿搿洋山芋丝再煸，哎呦搿洋山芋老脆呃老好吃，侬伐相信下趟去试试看]
  14. 激活=6.206  字=[师]  上下文=[老师，]  整句=[王老师，吾是搿能噶想呃，因为是辣辣一两年辰光刚刚普及了金融理财理财搿只物事]
  15. 激活=6.194  字=[开]  上下文=[老开开]  整句=[老开开心心呃，就放彻底放松，彻底享受生活]
  16. 激活=6.194  字=[嗯]  上下文=[嗯搿]  整句=[嗯搿吾相信呃]
  17. 激活=6.167  字=[去]  上下文=[去过]  整句=[去过呃地方嘛就]
  18. 激活=6.129  字=[能]  上下文=[搿能侪]  整句=[没撒呃，基本上搿能侪侪可以呃，就讲，反正反正这个，屋里相拧商量来大家做搿桩事体]
  19. 激活=6.098  字=[低]  上下文=[低价]  整句=[低价位呃股票嗯]
  20. 激活=6.098  字=[想]  上下文=[一想是]  整句=[后头来一想是搿种拧，伊讲，吧，就拿搿呃，伊拉，阿拉搿的里，迭呃拧后头做厂长唻]
  21. 激活=6.091  字=[葛]  上下文=[葛末]  整句=[葛末属于阿拉有撒事体]
  22. 激活=6.044  字=[是]  上下文=[讲是呃]  整句=[应该讲是呃伐大会的埃呃，因为伊拉开到现在几十年唻]
  23. 激活=6.028  字=[好]  上下文=[好几]  整句=[好几年唻，至少]
  24. 激活=6.020  字=[还]  上下文=[还可]  整句=[还可以呃，为撒道理，基本上没呢没撒坏呃]
  25. 激活=6.016  字=[葛]  上下文=[葛末]  整句=[葛末，后头文化大革命唻]
  26. 激活=5.981  字=[了]  上下文=[做了老]  整句=[葛末侬做了老好，侬搿单位，侬搿饭店开了三四年了，侬财务报表好拨吾看伐]
  27. 激活=5.976  字=[阿]  上下文=[，阿拉]  整句=[结果，阿拉单位里相头，开头吾老公单位里相头来了一呃拧]
  28. 激活=5.947  字=[煸]  上下文=[煸一]  整句=[煸一煸呢，然后侬拿搿洋山芋丝再煸，哎呦搿洋山芋老脆呃老好吃，侬伐相信下趟去试试看]
  29. 激活=5.932  字=[也]  上下文=[也是]  整句=[也是搿呃价钿咯]
  30. 激活=5.880  字=[进]  上下文=[进去]  整句=[进去，或者下半日就拿出来，也有呃，或者明朝拿出来，也有呃，伊搿只产品只有只利率低]

### Acoustic Features Statistics (from top-30 transcripts)
  - snr_db: mean=6.323, std=1.065, n=30
  - speech_rate: mean=19.373, std=12.088, n=30
  - pitch_mean: mean=184.914, std=46.526, n=30
  - pitch_std: mean=48.228, std=28.338, n=30
  - pitch_range: mean=201.245, std=109.638, n=30
  - energy_mean: mean=0.027, std=0.019, n=30
  - energy_std: mean=0.025, std=0.023, n=30
  - energy_dynamic_range: mean=40.855, std=5.567, n=30
  - silence_ratio: mean=0.698, std=0.107, n=30
  - speech_duration: mean=1.383, std=0.733, n=30
  - total_duration: mean=4.676, std=1.938, n=30
  - zero_crossing_rate: mean=0.144, std=0.046, n=30
  - spectral_centroid: mean=1889.879, std=370.557, n=30


### Linguistic Features Statistics (from top-30 transcripts)
  - char_count: mean=24.967, std=12.927
  - word_count: mean=1.000, std=0.000
  - contains_numbers: 0/30 (0.0% True)
  - contains_english: 0/30 (0.0% True)
  - contains_punctuation: 23/30 (76.7% True)
  - number_ratio: mean=0.000, std=0.000
  - contains_person_name: 2/30 (6.7% True)
  - contains_place_name: 3/30 (10.0% True)
  - contains_org_name: 2/30 (6.7% True)
  - voiced_initial_ratio: mean=0.417, std=0.103
  - nasal_final_ratio: mean=0.185, std=0.104


============================================================
REQUIRED OUTPUT FORMAT
============================================================

Return a JSON array with exactly 3 hypotheses:

[
  {
    "Reasoning": "<Detailed explanation referencing specific chars, contexts, and acoustic patterns>",
    "Phenomenon": "<Name of the speech/linguistic phenomenon>",
    "Confidence": "<1-4>"
  },
  ...
]

Confidence Scoring:
- 4: Character distribution + context patterns + acoustic trends all strongly align
- 3: Most evidence aligns, minor inconsistencies
- 2: Some evidence aligns, weaker support
- 1: Very little aligns, highly uncertain

============================================================
CHECKLIST FOR REASONING (address each item)
============================================================

1. **Character Pattern Analysis**:
   - Which characters dominate the distribution? Are they phonetically similar?
   - Do the context windows (aligned_context) reveal consistent phonetic neighbors?
   - Are there repeated syllable structures or tonal patterns across contexts?

2. **Context Window Interpretation**:
   - What characters appear before/after the aligned character most often?
   - Do context patterns suggest word boundaries, tone sandhi, or co-articulation effects?

3. **Acoustic Feature Interpretation**:
   - Do energy/pitch patterns suggest specific phonetic categories (voiced/unvoiced, tonal)?
   - Does spectral centroid or zero_crossing_rate indicate fricatives, stops, or nasals?

4. **Dialect-Specific Analysis**:
   - Are the high-frequency characters Shanghainese-specific vocabulary?
   - Do context patterns suggest Wu dialect tone sandhi rules?

5. **Linguistic Level**:
   - Is this a phoneme-level, syllable-level, word-level, or sentence-level feature?

6. **Error Pattern Correlation**:
   - Could this neuron's activation pattern relate to specific ASR confusion types?
   - Homophone errors? Tone errors? Word boundary errors?

======================================================================


======================================================================
You are given data from a Whisper-based Automatic Speech Recognition (ASR) model fine-tuned for Shanghainese (上海话) dialect.

Model Logic:
- Input: Raw audio waveform (Shanghainese speech)
- Encoder: Extracts acoustic features → hidden representations
- SAE Layer: Sparse Autoencoder applied to encoder layer 12 hidden states
- Decoder: Generates text transcription from encoded features
- Output: Chinese text transcription

The provided data describes a **concept** extracted by a Sparse Autoencoder (SAE).
A concept = a speech/linguistic feature detected from the audio that influences transcription output.

Your Task:
Using the character distribution, context windows, and acoustic/linguistic statistics below,
identify **three possible speech/linguistic phenomena** this neuron concept could represent,
ranked in descending order of confidence.

============================================================
DATA FORMAT DEFINITIONS
============================================================

1. **Character Distribution** (主线):
   The most frequent characters (aligned_char) among the top-500 highest-activation tokens.
   Each entry shows: character, occurrence count, mean activation value, and context window examples.
   The context window (aligned_context) is the ±1 character window around the aligned character
   in the transcript, giving local phonetic/lexical context.

2. **Top 30 Token Evidence** (具体证据):
   The 30 individual tokens with highest activation values.
   Each entry shows: activation value, the aligned character, its context window, and the full transcript.
   This preserves per-token detail so you can spot patterns not visible in aggregated statistics.

3. **Acoustic Features**: Audio-level statistics (energy, pitch, spectral centroid, etc.)
   computed over the full transcript of each top-activation token.

4. **Linguistic Features**: Text-level statistics (char count, punctuation, dialect markers, etc.)
   computed over the full transcript of each top-activation token.

============================================================
CURRENT CONCEPT DATA: Neuron #459
(Total activation events: 1669)
============================================================

### Character Distribution (Top 10 chars, from top-500 tokens)
   1. 字=[，]  出现=21次  平均激活=1.703  上下文示例: 呃，吾、诶，万、诶，就、是，去、唻，葛
   2. 字=[侬]  出现=20次  平均激活=1.842  上下文示例: 讲侬肉、侬讲、侬还、在侬，、侬哪
   3. 字=[诶]  出现=19次  平均激活=1.772  上下文示例: 诶，、诶诶
   4. 字=[呃]  出现=18次  平均激活=1.867  上下文示例: 真呃老、关呃传、撒呃，、真呃侪、拧呃关
   5. 字=[末]  出现=16次  平均激活=1.806  上下文示例: 葛末也、乃末侬、乃末嫁、葛末喏、葛末后
   6. 字=[伊]  出现=15次  平均激活=1.713  上下文示例: 是伊当、伊没、伊讲、伊还、伊拿
   7. 字=[也]  出现=14次  平均激活=1.871  上下文示例: 吾也做、也伐、也没、也要、是也蛮
   8. 字=[乃]  出现=14次  平均激活=1.727  上下文示例: 乃末、，乃末
   9. 字=[搿]  出现=12次  平均激活=1.756  上下文示例: 照搿方、搿还、搿呃、炎搿种、搿黄
  10. 字=[侪]  出现=11次  平均激活=1.957  上下文示例: 吾侪搭、家侪吃、侪伊、侪是

### Top 30 Token Evidence (highest activation, per-token detail)
   1. 激活=3.619  字=[辣]  上下文=[辣辣是]  整句=[辣辣是，一二年辰光伐是，一般性人家是，伐是做股票呃，侪进去孛相嘛]
   2. 激活=2.894  字=[辣]  上下文=[辣辣是]  整句=[辣辣是，一二年辰光伐是，一般性人家是，伐是做股票呃，侪进去孛相嘛]
   3. 激活=2.868  字=[辣]  上下文=[辣辣是]  整句=[辣辣是，一二年辰光伐是，一般性人家是，伐是做股票呃，侪进去孛相嘛]
   4. 激活=2.681  字=[呃]  上下文=[真呃老]  整句=[诶，真呃老节约伊讲，伊拉埃面外头撒呃奶茶咯，撒呃，诶，撒呃，诶种噶撒呃]
   5. 激活=2.669  字=[啊]  上下文=[拧啊真]  整句=[小拧啊真呃侪搿能介呃，特别男小拧还好，男大拧还好]
   6. 激活=2.668  字=[侪]  上下文=[吾侪搭]  整句=[吾侪搭界，小学里同学，中学里同学]
   7. 激活=2.623  字=[侬]  上下文=[讲侬肉]  整句=[伊就讲侬肉还是要少吃，肉只好吃搿能噶一只网球搿能噶大]
   8. 激活=2.608  字=[也]  上下文=[吾也做]  整句=[吾也做拨伊看吾对阿婆好吾也做摆阿拉新妇看]
   9. 激活=2.608  字=[对]  上下文=[对伐]  整句=[对伐，伐可以迭呃，侬养么样伊了]
  10. 激活=2.587  字=[吾]  上下文=[，吾是]  整句=[王老师，吾是搿能噶想呃，因为是辣辣一两年辰光刚刚普及了金融理财理财搿只物事]
  11. 激活=2.546  字=[了]  上下文=[觉了，]  整句=[每天夜到就来困觉了，早哴头就跑出去了，人家门一开伊就跟出去了，现在呐搿猫呐已经养了老大了]
  12. 激活=2.525  字=[要]  上下文=[烧要先]  整句=[红烧要先放调料，醋要最后加对吧]
  13. 激活=2.504  字=[吾]  上下文=[，吾是]  整句=[王老师，吾是搿能噶想呃，因为是辣辣一两年辰光刚刚普及了金融理财理财搿只物事]
  14. 激活=2.501  字=[呃]  上下文=[真呃老]  整句=[诶，真呃老节约伊讲，伊拉埃面外头撒呃奶茶咯，撒呃，诶，撒呃，诶种噶撒呃]
  15. 激活=2.438  字=[到]  上下文=[到，]  整句=[到，伊咯，伊老早]
  16. 激活=2.372  字=[吃]  上下文=[吃爷]  整句=[吃爷娘呃，伐像阿拉对爷娘侪介好]
  17. 激活=2.356  字=[末]  上下文=[葛末也]  整句=[葛末也会得选择，葛末也就伐应伐伐一定呢会得选星级宾馆]
  18. 激活=2.326  字=[吾]  上下文=[吾现]  整句=[吾现在打呃比方有的一千股]
  19. 激活=2.316  字=[伐]  上下文=[对伐，]  整句=[对伐，伐可以迭呃，侬养么样伊了]
  20. 激活=2.307  字=[上]  上下文=[本上搿]  整句=[没撒呃，基本上搿能侪侪可以呃，就讲，反正反正这个，屋里相拧商量来大家做搿桩事体]
  21. 激活=2.304  字=[家]  上下文=[人家是]  整句=[人家是讲调料调料，纯在调料]
  22. 激活=2.302  字=[起]  上下文=[起来]  整句=[起来，那娘的呃]
  23. 激活=2.289  字=[侬]  上下文=[侬讲]  整句=[侬讲是伐[+]]
  24. 激活=2.282  字=[乃]  上下文=[乃末]  整句=[乃末诶呃叫撒啊]
  25. 激活=2.271  字=[老]  上下文=[吾老早]  整句=[侬，吾老早一直是烧呃，吾是讲荤荤呃汤吾伐大考虑呃]
  26. 激活=2.269  字=[关]  上下文=[呃关系]  整句=[所以家庭呃拧呃关系也要，要搞好，自家呢，对自家阿哥啊，阿嫂啊，阿妹啊撒物事]
  27. 激活=2.256  字=[中]  上下文=[中哴]  整句=[中哴向吃，夜里向吃]
  28. 激活=2.252  字=[侪]  上下文=[家侪吃]  整句=[当然呢，大家侪吃了也侪老满意呃老放心呃，葛末也老省力呃，侬伐要撒呃，嗯]
  29. 激活=2.252  字=[呃]  上下文=[关呃传]  整句=[所以讲交关呃传统，交关呃优，作风，优良搿种呃传统呃作风，侪要传下去]
  30. 激活=2.247  字=[相]  上下文=[卖相伐]  整句=[葛末卖相伐好看，吃呃感觉还就伐一样，搿碧绿生青多少好辣，侬讲]

### Acoustic Features Statistics (from top-30 transcripts)
  - snr_db: mean=5.977, std=0.811, n=30
  - speech_rate: mean=19.916, std=7.460, n=30
  - pitch_mean: mean=179.131, std=49.983, n=30
  - pitch_std: mean=40.090, std=18.769, n=30
  - pitch_range: mean=171.360, std=87.934, n=30
  - energy_mean: mean=0.025, std=0.025, n=30
  - energy_std: mean=0.023, std=0.027, n=30
  - energy_dynamic_range: mean=39.282, std=6.497, n=30
  - silence_ratio: mean=0.712, std=0.117, n=30
  - speech_duration: mean=1.192, std=0.627, n=30
  - total_duration: mean=4.533, std=2.210, n=30
  - zero_crossing_rate: mean=0.142, std=0.044, n=30
  - spectral_centroid: mean=1861.189, std=413.938, n=30


### Linguistic Features Statistics (from top-30 transcripts)
  - char_count: mean=23.900, std=11.297
  - word_count: mean=1.000, std=0.000
  - contains_numbers: 0/30 (0.0% True)
  - contains_english: 0/30 (0.0% True)
  - contains_punctuation: 26/30 (86.7% True)
  - number_ratio: mean=0.000, std=0.000
  - contains_person_name: 2/30 (6.7% True)
  - contains_place_name: 1/30 (3.3% True)
  - contains_org_name: 0/30 (0.0% True)
  - voiced_initial_ratio: mean=0.404, std=0.133
  - nasal_final_ratio: mean=0.196, std=0.115


============================================================
REQUIRED OUTPUT FORMAT
============================================================

Return a JSON array with exactly 3 hypotheses:

[
  {
    "Reasoning": "<Detailed explanation referencing specific chars, contexts, and acoustic patterns>",
    "Phenomenon": "<Name of the speech/linguistic phenomenon>",
    "Confidence": "<1-4>"
  },
  ...
]

Confidence Scoring:
- 4: Character distribution + context patterns + acoustic trends all strongly align
- 3: Most evidence aligns, minor inconsistencies
- 2: Some evidence aligns, weaker support
- 1: Very little aligns, highly uncertain

============================================================
CHECKLIST FOR REASONING (address each item)
============================================================

1. **Character Pattern Analysis**:
   - Which characters dominate the distribution? Are they phonetically similar?
   - Do the context windows (aligned_context) reveal consistent phonetic neighbors?
   - Are there repeated syllable structures or tonal patterns across contexts?

2. **Context Window Interpretation**:
   - What characters appear before/after the aligned character most often?
   - Do context patterns suggest word boundaries, tone sandhi, or co-articulation effects?

3. **Acoustic Feature Interpretation**:
   - Do energy/pitch patterns suggest specific phonetic categories (voiced/unvoiced, tonal)?
   - Does spectral centroid or zero_crossing_rate indicate fricatives, stops, or nasals?

4. **Dialect-Specific Analysis**:
   - Are the high-frequency characters Shanghainese-specific vocabulary?
   - Do context patterns suggest Wu dialect tone sandhi rules?

5. **Linguistic Level**:
   - Is this a phoneme-level, syllable-level, word-level, or sentence-level feature?

6. **Error Pattern Correlation**:
   - Could this neuron's activation pattern relate to specific ASR confusion types?
   - Homophone errors? Tone errors? Word boundary errors?

======================================================================

