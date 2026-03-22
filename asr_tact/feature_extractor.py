"""
ASR Feature Extractor

提取音频样本的多维度特征，用于神经元语义标注
包含三层特征：声学特征、语言特征、错误模式特征
"""

import torch
import numpy as np
import re
import json
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field, asdict
from collections import Counter
import librosa


@dataclass
class AcousticFeatures:
    """
    声学特征
    
    特征定义:
    - snr_db: 信噪比 (dB)，使用 WADA-SNR 算法估计
    - snr_level: SNR 等级 (low: <10dB, medium: 10-20dB, high: >20dB)
    - speech_rate: 语速 (字/秒)，基于有效语音时长计算
    - speech_rate_level: 语速等级 (slow: <3, normal: 3-6, fast: >6)
    - pitch_mean: 基频均值 (Hz)，使用 pYIN 算法提取
    - pitch_std: 基频标准差 (Hz)
    - pitch_range: 基频范围 (Hz)，max - min
    - pitch_contour: 音高轮廓类型 (rising/falling/flat/varied/question)
    - energy_mean: RMS 能量均值
    - energy_std: RMS 能量标准差
    - energy_dynamic_range: 动态范围 (dB)
    - silence_ratio: 静音比例 (基于 VAD)
    - speech_duration: 有效语音时长 (秒)
    - total_duration: 总时长 (秒)
    - zero_crossing_rate: 过零率均值
    - spectral_centroid: 频谱质心均值 (Hz)
    """
    # 信噪比相关
    snr_db: float = 0.0
    snr_level: str = "unknown"  # low/medium/high
    
    # 语速相关
    speech_rate: float = 0.0  # 字/秒 (基于有效语音时长)
    speech_rate_level: str = "unknown"  # slow/normal/fast
    
    # 音高相关 (基频 F0)
    pitch_mean: float = 0.0  # Hz
    pitch_std: float = 0.0  # Hz
    pitch_range: float = 0.0  # Hz (max - min)
    pitch_contour: str = "unknown"  # rising/falling/flat/varied/question
    
    # 能量相关
    energy_mean: float = 0.0  # RMS
    energy_std: float = 0.0
    energy_dynamic_range: float = 0.0  # dB
    
    # 时长相关
    silence_ratio: float = 0.0  # 静音比例
    speech_duration: float = 0.0  # 有效语音时长 (秒)
    total_duration: float = 0.0  # 总时长 (秒)
    
    # 频谱特征
    zero_crossing_rate: float = 0.0  # 过零率
    spectral_centroid: float = 0.0  # 频谱质心 (Hz)
    
    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class LinguisticFeatures:
    """
    语言特征
    
    特征定义:
    - transcript: 原始文本
    - char_count: 字符数（去除空格）
    - word_count: 词数（按空格分词）
    - contains_numbers: 是否包含数字
    - number_ratio: 数字占比
    - contains_english: 是否包含英文
    - contains_punctuation: 是否包含标点
    - contains_person_name: 是否包含人名
    - contains_place_name: 是否包含地名
    - contains_org_name: 是否包含机构名
    - domain: 领域 (general/medical/legal/tech/finance)
    - domain_keywords: 匹配到的领域关键词
    - sentence_type: 句型 (declarative/interrogative/imperative/exclamatory)
    - dialect_markers: 方言标记词列表
    - initial_distribution: 声母分布 (上海话声母系统)
    - final_distribution: 韵母分布 (上海话韵母系统)
    - tone_pattern: 声调模式
    """
    # 文本内容
    transcript: str = ""
    
    # 基本统计
    char_count: int = 0
    word_count: int = 0
    
    # 词汇特征
    contains_numbers: bool = False
    contains_english: bool = False
    contains_punctuation: bool = False
    number_ratio: float = 0.0
    
    # 命名实体
    contains_person_name: bool = False
    contains_place_name: bool = False
    contains_org_name: bool = False
    
    # 领域
    domain: str = "general"  # general/medical/legal/tech/finance
    domain_keywords: List[str] = field(default_factory=list)
    
    # 句型
    sentence_type: str = "declarative"  # declarative/interrogative/imperative/exclamatory
    
    # 方言特征
    dialect_markers: List[str] = field(default_factory=list)
    
    # 上海话声韵母分布
    initial_distribution: Dict[str, int] = field(default_factory=dict)  # 声母分布
    final_distribution: Dict[str, int] = field(default_factory=dict)    # 韵母分布
    voiced_initial_ratio: float = 0.0  # 浊辅音比例 (上海话特色)
    nasal_final_ratio: float = 0.0     # 鼻韵母比例
    
    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ErrorPatternFeatures:
    """错误模式特征"""
    # 预测结果
    prediction: str = ""
    ground_truth: str = ""
    
    # 错误统计
    cer: float = 0.0  # 字符错误率
    wer: float = 0.0  # 词错误率
    
    # 错误类型
    substitution_count: int = 0
    insertion_count: int = 0
    deletion_count: int = 0
    
    # 错误模式
    error_type: str = "none"  # none/homophone/boundary/oov/noise/dialect
    homophone_errors: List[Tuple[str, str]] = field(default_factory=list)  # (预测, 正确)
    boundary_errors: List[str] = field(default_factory=list)
    oov_words: List[str] = field(default_factory=list)
    
    def to_dict(self) -> dict:
        result = asdict(self)
        # 转换 tuple 列表为可序列化格式
        result['homophone_errors'] = [list(x) for x in self.homophone_errors]
        return result


@dataclass
class SampleFeatures:
    """样本完整特征"""
    sample_id: str = ""
    audio_path: str = ""
    
    acoustic: AcousticFeatures = field(default_factory=AcousticFeatures)
    linguistic: LinguisticFeatures = field(default_factory=LinguisticFeatures)
    error_pattern: ErrorPatternFeatures = field(default_factory=ErrorPatternFeatures)
    
    # 元数据
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> dict:
        return {
            'sample_id': self.sample_id,
            'audio_path': self.audio_path,
            'acoustic': self.acoustic.to_dict(),
            'linguistic': self.linguistic.to_dict(),
            'error_pattern': self.error_pattern.to_dict(),
            'metadata': self.metadata,
        }
    
    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2)


class ASRFeatureExtractor:
    """
    ASR 特征提取器
    
    从音频和文本中提取多维度特征，用于神经元语义分析
    """
    
    # 常见同音字对 (上海话/普通话)
    HOMOPHONE_PAIRS = {
        # 普通话常见混淆
        ('的', '地'), ('的', '得'), ('地', '得'),
        ('在', '再'), ('做', '作'),
        ('他', '她'), ('他', '它'),
        ('那', '哪'), ('呢', '呐'),
        # 上海话特有
        ('侬', '农'), ('阿', '啊'),
        ('伊', '一'), ('搿', '个'),
        ('勿', '物'), ('啥', '撒'),
    }
    
    # 领域关键词
    DOMAIN_KEYWORDS = {
        'medical': ['医院', '医生', '病人', '手术', '药', '治疗', '检查', '诊断', '症状', '病'],
        'legal': ['法院', '律师', '案件', '判决', '法律', '合同', '诉讼', '被告', '原告', '证据'],
        'tech': ['代码', '程序', '软件', '系统', '数据', '网络', '服务器', '算法', '接口', '开发'],
        'finance': ['银行', '股票', '基金', '投资', '贷款', '利率', '账户', '交易', '金融', '理财'],
    }
    
    # 上海话特征词
    SHANGHAI_MARKERS = [
        '侬', '阿拉', '伊', '搿', '勿', '啥', '哪能', '老', '交关', '蛮',
        '覅', '来三', '戆', '瘪三', '拎勿清', '门槛精', '噶', '嘎',
    ]
    
    # 上海话声韵母分析器 (延迟加载)
    _shanghai_analyzer = None
    
    @classmethod
    def _get_shanghai_analyzer(cls):
        """获取上海话声韵母分析器 (延迟加载)"""
        if cls._shanghai_analyzer is None:
            try:
                from .shanghai_phoneme import ShanghaiPhonemeAnalyzer
                cls._shanghai_analyzer = ShanghaiPhonemeAnalyzer()
            except Exception as e:
                print(f"Warning: Failed to load Shanghai phoneme analyzer: {e}")
                cls._shanghai_analyzer = False  # 标记为加载失败
        return cls._shanghai_analyzer if cls._shanghai_analyzer else None
    
    def __init__(self, sample_rate: int = 16000):
        self.sample_rate = sample_rate
    
    def extract_features(
        self,
        audio: Optional[np.ndarray] = None,
        audio_path: Optional[str] = None,
        transcript: str = "",
        prediction: str = "",
        sample_id: str = "",
        metadata: Optional[Dict] = None,
    ) -> SampleFeatures:
        """
        提取样本的完整特征
        
        Args:
            audio: 音频波形数组
            audio_path: 音频文件路径
            transcript: 真实转录文本
            prediction: 模型预测文本
            sample_id: 样本ID
            metadata: 额外元数据
            
        Returns:
            SampleFeatures: 完整特征对象
        """
        features = SampleFeatures(
            sample_id=sample_id,
            audio_path=audio_path or "",
            metadata=metadata or {},
        )
        
        # 加载音频
        if audio is None and audio_path:
            try:
                audio, _ = librosa.load(audio_path, sr=self.sample_rate)
            except Exception as e:
                print(f"Warning: Failed to load audio {audio_path}: {e}")
                audio = None
        
        # 提取声学特征
        if audio is not None:
            features.acoustic = self._extract_acoustic_features(audio, transcript)
        
        # 提取语言特征
        if transcript:
            features.linguistic = self._extract_linguistic_features(transcript)
        
        # 提取错误模式特征
        if transcript and prediction:
            features.error_pattern = self._extract_error_features(prediction, transcript)
        
        return features
    
    def _extract_acoustic_features(
        self, 
        audio: np.ndarray,
        transcript: str = ""
    ) -> AcousticFeatures:
        """
        提取声学特征
        
        使用的算法:
        - SNR: WADA-SNR (Waveform Amplitude Distribution Analysis)
        - VAD: 基于能量和过零率的语音活动检测
        - F0: pYIN 算法 (概率 YIN)
        - 语速: 字符数 / 有效语音时长
        """
        features = AcousticFeatures()
        
        # 总时长
        features.total_duration = len(audio) / self.sample_rate
        
        # ========== 1. 语音活动检测 (VAD) ==========
        # 使用能量和过零率联合判断
        frame_length = int(0.025 * self.sample_rate)  # 25ms 帧
        hop_length = int(0.010 * self.sample_rate)    # 10ms 帧移
        
        # 计算短时能量 (RMS)
        rms = librosa.feature.rms(y=audio, frame_length=frame_length, hop_length=hop_length)[0]
        
        # 计算过零率
        
        # VAD: 能量 > 阈值 且 过零率在合理范围内 (语音通常 0.02-0.2)
        # 阈值: 使用能量的 Otsu 自适应阈值
        energy_threshold = self._otsu_threshold(rms)
        speech_frames = (rms > energy_threshold) & (zcr > 0.01) & (zcr < 0.3)
        
        # 静音比例和有效语音时长
        features.silence_ratio = float(1 - np.mean(speech_frames))
        features.speech_duration = float(np.sum(speech_frames) * hop_length / self.sample_rate)
        
        # ========== 2. 信噪比估计 (WADA-SNR) ==========
        features.snr_db = self._estimate_wada_snr(audio, speech_frames, rms, hop_length)
        if features.snr_db < 10:
            features.snr_level = "low"
        elif features.snr_db < 20:
            features.snr_level = "medium"
        else:
            features.snr_level = "high"
        
        # ========== 3. 语速计算 ==========
        if transcript and features.speech_duration > 0.1:
            char_count = len(transcript.replace(" ", "").replace("，", "").replace("。", ""))
            features.speech_rate = char_count / features.speech_duration
            if features.speech_rate < 3:
                features.speech_rate_level = "slow"
            elif features.speech_rate < 6:
                features.speech_rate_level = "normal"
            else:
                features.speech_rate_level = "fast"
        
        # ========== 4. 基频 (F0) 分析 ==========
        try:
            # 使用 pYIN 算法提取基频 (更准确)
            f0, voiced_flag, voiced_probs = librosa.pyin(
                audio,
                fmin=librosa.note_to_hz('C2'),  # ~65 Hz
                fmax=librosa.note_to_hz('C7'),  # ~2093 Hz
                sr=self.sample_rate,
                frame_length=frame_length * 4,  # pYIN 需要更长的帧
            )
            
            # 只取有声段的 F0
            valid_f0 = f0[~np.isnan(f0)]
            
            if len(valid_f0) > 5:
                features.pitch_mean = float(np.mean(valid_f0))
                features.pitch_std = float(np.std(valid_f0))
                features.pitch_range = float(np.max(valid_f0) - np.min(valid_f0))
                
                # 音高轮廓分析
                features.pitch_contour = self._analyze_pitch_contour(valid_f0)
        except Exception:
            pass
        
        # ========== 5. 能量特征 ==========
        features.energy_mean = float(np.mean(rms))
        features.energy_std = float(np.std(rms))
        
        # 动态范围 (dB)
        rms_nonzero = rms[rms > 1e-10]
        if len(rms_nonzero) > 0:
            max_db = 20 * np.log10(np.max(rms_nonzero))
            min_db = 20 * np.log10(np.percentile(rms_nonzero, 5))  # 5% 分位数避免极端值
            features.energy_dynamic_range = float(max_db - min_db)
        
        # ========== 6. 频谱特征 ==========
        features.zero_crossing_rate = float(np.mean(zcr))
        
        # 频谱质心
        spectral_centroids = librosa.feature.spectral_centroid(
            y=audio, sr=self.sample_rate, 
            n_fft=frame_length, hop_length=hop_length
        )[0]
        features.spectral_centroid = float(np.mean(spectral_centroids))
        
        return features
    
    def _otsu_threshold(self, values: np.ndarray) -> float:
        """
        Otsu 自适应阈值算法
        用于自动确定能量阈值，区分语音和静音
        """
        # 归一化到 0-255
        values_norm = values - values.min()
        if values_norm.max() > 0:
            values_norm = (values_norm / values_norm.max() * 255).astype(np.uint8)
        else:
            return 0.0
        
        # 计算直方图
        hist, bin_edges = np.histogram(values_norm, bins=256, range=(0, 256))
        hist = hist.astype(float) / hist.sum()
        
        # Otsu 算法
        best_threshold = 0
        best_variance = 0
        
        for t in range(1, 256):
            w0 = hist[:t].sum()
            w1 = hist[t:].sum()
            
            if w0 == 0 or w1 == 0:
                continue
            
            mu0 = np.sum(np.arange(t) * hist[:t]) / w0
            mu1 = np.sum(np.arange(t, 256) * hist[t:]) / w1
            
            variance = w0 * w1 * (mu0 - mu1) ** 2
            
            if variance > best_variance:
                best_variance = variance
                best_threshold = t
        
        # 转换回原始尺度
        return values.min() + (best_threshold / 255) * (values.max() - values.min())
    
    def _estimate_wada_snr(
        self, 
        audio: np.ndarray, 
        speech_frames: np.ndarray,
        rms: np.ndarray,
        hop_length: int
    ) -> float:
        """
        WADA-SNR 估计 (Waveform Amplitude Distribution Analysis)
        
        原理:
        - 语音段的能量分布与噪声段不同
        - 通过比较语音段和非语音段的能量来估计 SNR
        
        Args:
            audio: 音频波形
            speech_frames: VAD 结果 (布尔数组)
            rms: 每帧的 RMS 能量
            hop_length: 帧移
        
        Returns:
            SNR (dB)
        """
        if len(speech_frames) == 0 or np.sum(speech_frames) == 0:
            return 20.0  # 默认值
        
        # 分离语音帧和噪声帧的能量
        speech_energy = rms[speech_frames]
        noise_frames = ~speech_frames
        
        if np.sum(noise_frames) < 3:
            # 噪声帧太少，使用语音帧的最低能量作为噪声估计
            noise_energy = np.percentile(speech_energy, 10)
        else:
            noise_energy = np.mean(rms[noise_frames])
        
        signal_energy = np.mean(speech_energy)
        
        # 计算 SNR
        if noise_energy > 1e-10:
            snr = 10 * np.log10(signal_energy / noise_energy + 1e-10)
            return float(np.clip(snr, 0, 50))
        
        return 30.0  # 噪声极低时返回高 SNR
    
    def _analyze_pitch_contour(self, f0_values: np.ndarray) -> str:
        """
        分析音高轮廓类型
        
        方法:
        1. 将 F0 序列分成 4 段，计算每段均值
        2. 分析趋势：上升、下降、平坦、变化、疑问句（先降后升）
        
        Args:
            f0_values: 有效的 F0 值序列
        
        Returns:
            轮廓类型: rising/falling/flat/varied/question
        """
        n = len(f0_values)
        if n < 8:
            return "unknown"
        
        # 分成 4 段
        segment_size = n // 4
        segments = [
            np.mean(f0_values[i*segment_size:(i+1)*segment_size])
            for i in range(4)
        ]
        
        # 计算变化率
        mean_f0 = np.mean(f0_values)
        std_f0 = np.std(f0_values)
        cv = std_f0 / mean_f0 if mean_f0 > 0 else 0  # 变异系数
        
        # 判断轮廓类型
        if cv < 0.05:
            return "flat"  # 变化很小
        
        # 计算整体趋势
        first_half = np.mean(segments[:2])
        second_half = np.mean(segments[2:])
        overall_change = (second_half - first_half) / mean_f0
        
        # 检测疑问句模式：先降后升
        if segments[2] < segments[0] * 0.9 and segments[3] > segments[2] * 1.1:
            return "question"
        
        if overall_change > 0.1:
            return "rising"
        elif overall_change < -0.1:
            return "falling"
        elif cv > 0.15:
            return "varied"
        else:
            return "flat"
    
    def _extract_linguistic_features(self, transcript: str) -> LinguisticFeatures:
        """提取语言特征"""
        features = LinguisticFeatures()
        features.transcript = transcript
        
        # 基本统计
        features.char_count = len(transcript.replace(" ", ""))
        features.word_count = len(transcript.split())
        
        # 数字检测
        numbers = re.findall(r'\d+', transcript)
        features.contains_numbers = len(numbers) > 0
        if features.char_count > 0:
            features.number_ratio = sum(len(n) for n in numbers) / features.char_count
        
        # 英文检测
        features.contains_english = bool(re.search(r'[a-zA-Z]', transcript))
        
        # 标点检测
        features.contains_punctuation = bool(re.search(r'[，。！？、；：""''【】]', transcript))
        
        # 命名实体检测 (简单规则)
        features.contains_person_name = self._detect_person_name(transcript)
        features.contains_place_name = self._detect_place_name(transcript)
        features.contains_org_name = self._detect_org_name(transcript)
        
        # 领域检测
        features.domain, features.domain_keywords = self._detect_domain(transcript)
        
        # 句型检测
        features.sentence_type = self._detect_sentence_type(transcript)
        
        # 方言标记检测
        features.dialect_markers = self._detect_dialect_markers(transcript)
        
        # 上海话声韵母分析 (基于 RIME 吴语词库)
        phoneme_analysis = self._analyze_shanghai_phonemes(transcript)
        features.initial_distribution = phoneme_analysis['initial_distribution']
        features.final_distribution = phoneme_analysis['final_distribution']
        features.voiced_initial_ratio = phoneme_analysis['voiced_initial_ratio']
        features.nasal_final_ratio = phoneme_analysis['nasal_final_ratio']
        features.checked_final_ratio = phoneme_analysis.get('checked_final_ratio', 0.0)
        features.phoneme_coverage = phoneme_analysis.get('coverage', 0.0)
        
        return features
    
    def _analyze_shanghai_phonemes(self, text: str) -> Dict[str, Any]:
        """
        分析上海话声韵母分布
        
        使用 RIME 吴语上海话词库 (wugniu_zaonhe.dict.yaml) 进行分析
        
        吴语拉丁式注音法 (法吴) 声母系统:
        - 清塞音/塞擦音: p, ph, t, th, k, kh, ts, tsh, c, ch
        - 浊塞音/塞擦音: b, d, g, dz, j
        - 清擦音: f, s, sh, h
        - 浊擦音: v, z, zh, gh
        - 鼻音: m, n, ng, gn (ny), nk
        - 边音: l
        
        韵母系统:
        - 入声韵: 以 q 结尾 (aq, eq, iq, oq, uq, yq)
        - 鼻韵母: 以 n/ng 结尾
        
        Returns:
            Dict 包含:
            - initial_distribution: 声母分布统计
            - final_distribution: 韵母分布统计
            - voiced_initial_ratio: 浊辅音声母比例
            - nasal_final_ratio: 鼻韵母比例
            - checked_final_ratio: 入声韵比例 (上海话特色)
        """
        analyzer = self._get_shanghai_analyzer()
        
        if analyzer is not None:
            # 使用 RIME 词库分析
            result = analyzer.analyze_text(text)
            return {
                'initial_distribution': result['initial_distribution'],
                'final_distribution': result['final_distribution'],
                'voiced_initial_ratio': result['voiced_initial_ratio'],
                'nasal_final_ratio': result['nasal_final_ratio'],
                'checked_final_ratio': result.get('checked_final_ratio', 0.0),
                'coverage': result.get('coverage', 0.0),
            }
        else:
            # 词库未加载，返回空结果
            return {
                'initial_distribution': {},
                'final_distribution': {},
                'voiced_initial_ratio': 0.0,
                'nasal_final_ratio': 0.0,
                'checked_final_ratio': 0.0,
                'coverage': 0.0,
            }
    
    def _extract_error_features(
        self, 
        prediction: str, 
        ground_truth: str
    ) -> ErrorPatternFeatures:
        """提取错误模式特征"""
        features = ErrorPatternFeatures()
        features.prediction = prediction
        features.ground_truth = ground_truth
        
        # 计算 CER
        features.cer = self._compute_cer(prediction, ground_truth)
        
        # 计算编辑距离详情
        sub, ins, dele = self._compute_edit_details(prediction, ground_truth)
        features.substitution_count = sub
        features.insertion_count = ins
        features.deletion_count = dele
        
        # 检测同音字错误
        features.homophone_errors = self._detect_homophone_errors(prediction, ground_truth)
        
        # 检测边界错误
        features.boundary_errors = self._detect_boundary_errors(prediction, ground_truth)
        
        # 判断主要错误类型
        if features.cer == 0:
            features.error_type = "none"
        elif len(features.homophone_errors) > 0:
            features.error_type = "homophone"
        elif len(features.boundary_errors) > 0:
            features.error_type = "boundary"
        else:
            features.error_type = "other"
        
        return features
    
    
    def _detect_person_name(self, text: str) -> bool:
        """检测人名"""
        # 常见姓氏
        surnames = ['张', '王', '李', '赵', '刘', '陈', '杨', '黄', '周', '吴',
                   '徐', '孙', '马', '朱', '胡', '郭', '何', '林', '罗', '高']
        for surname in surnames:
            if surname in text:
                idx = text.index(surname)
                if idx + 2 <= len(text):
                    return True
        return False
    
    def _detect_place_name(self, text: str) -> bool:
        """检测地名"""
        place_markers = ['省', '市', '区', '县', '镇', '村', '路', '街', '大道', '广场',
                        '上海', '北京', '广州', '深圳', '杭州', '南京', '苏州']
        return any(marker in text for marker in place_markers)
    
    def _detect_org_name(self, text: str) -> bool:
        """检测机构名"""
        org_markers = ['公司', '集团', '银行', '医院', '学校', '大学', '学院', '研究院',
                      '政府', '局', '部', '委员会', '协会']
        return any(marker in text for marker in org_markers)
    
    def _detect_domain(self, text: str) -> Tuple[str, List[str]]:
        """检测领域"""
        domain_scores = {}
        domain_matches = {}
        
        for domain, keywords in self.DOMAIN_KEYWORDS.items():
            matches = [kw for kw in keywords if kw in text]
            domain_scores[domain] = len(matches)
            domain_matches[domain] = matches
        
        if max(domain_scores.values()) > 0:
            best_domain = max(domain_scores, key=domain_scores.get)
            return best_domain, domain_matches[best_domain]
        
        return "general", []
    
    def _detect_sentence_type(self, text: str) -> str:
        """检测句型"""
        if text.endswith('？') or text.endswith('?'):
            return "interrogative"
        elif text.endswith('！') or text.endswith('!'):
            return "exclamatory"
        elif any(word in text for word in ['请', '别', '不要', '必须']):
            return "imperative"
        else:
            return "declarative"
    
    def _detect_dialect_markers(self, text: str) -> List[str]:
        """检测方言标记"""
        return [marker for marker in self.SHANGHAI_MARKERS if marker in text]
    
    def _compute_cer(self, prediction: str, ground_truth: str) -> float:
        """计算字符错误率"""
        pred_chars = list(prediction.replace(" ", ""))
        ref_chars = list(ground_truth.replace(" ", ""))
        
        if len(ref_chars) == 0:
            return 0.0 if len(pred_chars) == 0 else 1.0
        
        # 动态规划计算编辑距离
        m, n = len(pred_chars), len(ref_chars)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        
        for i in range(m + 1):
            dp[i][0] = i
        for j in range(n + 1):
            dp[0][j] = j
        
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if pred_chars[i-1] == ref_chars[j-1]:
                    dp[i][j] = dp[i-1][j-1]
                else:
                    dp[i][j] = min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1]) + 1
        
        return dp[m][n] / len(ref_chars)
    
    def _compute_edit_details(
        self, 
        prediction: str, 
        ground_truth: str
    ) -> Tuple[int, int, int]:
        """计算编辑操作详情"""
        pred_chars = list(prediction.replace(" ", ""))
        ref_chars = list(ground_truth.replace(" ", ""))
        
        m, n = len(pred_chars), len(ref_chars)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        ops = [[None] * (n + 1) for _ in range(m + 1)]
        
        for i in range(m + 1):
            dp[i][0] = i
            if i > 0:
                ops[i][0] = 'I'
        for j in range(n + 1):
            dp[0][j] = j
            if j > 0:
                ops[0][j] = 'D'
        
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if pred_chars[i-1] == ref_chars[j-1]:
                    dp[i][j] = dp[i-1][j-1]
                    ops[i][j] = 'M'
                else:
                    costs = [
                        (dp[i-1][j-1] + 1, 'S'),
                        (dp[i-1][j] + 1, 'I'),
                        (dp[i][j-1] + 1, 'D'),
                    ]
                    dp[i][j], ops[i][j] = min(costs, key=lambda x: x[0])
        
        # 回溯统计
        sub, ins, dele = 0, 0, 0
        i, j = m, n
        while i > 0 or j > 0:
            op = ops[i][j]
            if op == 'M':
                i -= 1
                j -= 1
            elif op == 'S':
                sub += 1
                i -= 1
                j -= 1
            elif op == 'I':
                ins += 1
                i -= 1
            elif op == 'D':
                dele += 1
                j -= 1
            else:
                break
        
        return sub, ins, dele
    
    def _detect_homophone_errors(
        self, 
        prediction: str, 
        ground_truth: str
    ) -> List[Tuple[str, str]]:
        """检测同音字错误"""
        errors = []
        pred_chars = list(prediction.replace(" ", ""))
        ref_chars = list(ground_truth.replace(" ", ""))
        
        # 简单对齐检测
        min_len = min(len(pred_chars), len(ref_chars))
        for i in range(min_len):
            if pred_chars[i] != ref_chars[i]:
                pair = (pred_chars[i], ref_chars[i])
                reverse_pair = (ref_chars[i], pred_chars[i])
                if pair in self.HOMOPHONE_PAIRS or reverse_pair in self.HOMOPHONE_PAIRS:
                    errors.append(pair)
        
        return errors
    
    def _detect_boundary_errors(
        self, 
        prediction: str, 
        ground_truth: str
    ) -> List[str]:
        """检测边界错误（分词错误）"""
        errors = []
        
        # 检测连读/分词错误
        pred_no_space = prediction.replace(" ", "")
        ref_no_space = ground_truth.replace(" ", "")
        
        if pred_no_space == ref_no_space and prediction != ground_truth:
            errors.append("spacing_difference")
        
        return errors
    
    def batch_extract(
        self,
        samples: List[Dict],
        audio_key: str = "audio",
        transcript_key: str = "text",
        prediction_key: str = "prediction",
    ) -> List[SampleFeatures]:
        """批量提取特征"""
        results = []
        for i, sample in enumerate(samples):
            audio = sample.get(audio_key)
            if isinstance(audio, dict) and 'array' in audio:
                audio = audio['array']
            
            features = self.extract_features(
                audio=audio,
                audio_path=sample.get('audio_path', sample.get('path', '')),
                transcript=sample.get(transcript_key, ''),
                prediction=sample.get(prediction_key, ''),
                sample_id=str(sample.get('id', i)),
                metadata=sample.get('metadata', {}),
            )
            results.append(features)
        
        return results
    
    def save_features(
        self, 
        features: List[SampleFeatures], 
        output_path: str
    ):
        """保存特征到 JSON 文件"""
        data = [f.to_dict() for f in features]
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    
    @staticmethod
    def load_features(input_path: str) -> List[Dict]:
        """从 JSON 文件加载特征"""
        with open(input_path, 'r', encoding='utf-8') as f:
            return json.load(f)
