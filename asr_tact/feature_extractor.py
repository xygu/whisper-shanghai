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
    """声学特征"""
    # 信噪比相关
    snr_db: float = 0.0
    snr_level: str = "unknown"  # low/medium/high
    
    # 语速相关
    speech_rate: float = 0.0  # 字/秒
    speech_rate_level: str = "unknown"  # slow/normal/fast
    
    # 音高相关
    pitch_mean: float = 0.0
    pitch_std: float = 0.0
    pitch_contour: str = "unknown"  # rising/falling/flat/varied
    
    # 能量相关
    energy_mean: float = 0.0
    energy_std: float = 0.0
    silence_ratio: float = 0.0  # 静音比例
    
    # 时长
    duration_seconds: float = 0.0
    
    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class LinguisticFeatures:
    """语言特征"""
    # 文本内容
    transcript: str = ""
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
    
    # 领域特征
    domain: str = "general"  # general/medical/legal/tech/finance
    domain_keywords: List[str] = field(default_factory=list)
    
    # 句法特征
    sentence_type: str = "declarative"  # declarative/interrogative/imperative/exclamatory
    
    # 方言特征 (针对上海话)
    dialect_markers: List[str] = field(default_factory=list)
    
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
        """提取声学特征"""
        features = AcousticFeatures()
        
        # 时长
        features.duration_seconds = len(audio) / self.sample_rate
        
        # 信噪比估计
        features.snr_db = self._estimate_snr(audio)
        if features.snr_db < 10:
            features.snr_level = "low"
        elif features.snr_db < 20:
            features.snr_level = "medium"
        else:
            features.snr_level = "high"
        
        # 语速估计
        if transcript and features.duration_seconds > 0:
            char_count = len(transcript.replace(" ", ""))
            features.speech_rate = char_count / features.duration_seconds
            if features.speech_rate < 3:
                features.speech_rate_level = "slow"
            elif features.speech_rate < 6:
                features.speech_rate_level = "normal"
            else:
                features.speech_rate_level = "fast"
        
        # 音高分析
        try:
            pitches, magnitudes = librosa.piptrack(y=audio, sr=self.sample_rate)
            pitch_values = []
            for t in range(pitches.shape[1]):
                index = magnitudes[:, t].argmax()
                pitch = pitches[index, t]
                if pitch > 0:
                    pitch_values.append(pitch)
            
            if pitch_values:
                features.pitch_mean = float(np.mean(pitch_values))
                features.pitch_std = float(np.std(pitch_values))
                
                # 判断音高轮廓
                if len(pitch_values) > 10:
                    first_half = np.mean(pitch_values[:len(pitch_values)//2])
                    second_half = np.mean(pitch_values[len(pitch_values)//2:])
                    diff = second_half - first_half
                    
                    if features.pitch_std < features.pitch_mean * 0.1:
                        features.pitch_contour = "flat"
                    elif diff > features.pitch_mean * 0.1:
                        features.pitch_contour = "rising"
                    elif diff < -features.pitch_mean * 0.1:
                        features.pitch_contour = "falling"
                    else:
                        features.pitch_contour = "varied"
        except Exception:
            pass
        
        # 能量分析
        features.energy_mean = float(np.mean(np.abs(audio)))
        features.energy_std = float(np.std(np.abs(audio)))
        
        # 静音比例
        silence_threshold = 0.01 * np.max(np.abs(audio))
        features.silence_ratio = float(np.mean(np.abs(audio) < silence_threshold))
        
        return features
    
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
        
        return features
    
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
    
    def _estimate_snr(self, audio: np.ndarray) -> float:
        """估计信噪比"""
        # 简单的 SNR 估计：基于能量分布
        frame_length = int(0.025 * self.sample_rate)
        hop_length = int(0.010 * self.sample_rate)
        
        # 计算帧能量
        frames = librosa.util.frame(audio, frame_length=frame_length, hop_length=hop_length)
        frame_energy = np.sum(frames ** 2, axis=0)
        
        if len(frame_energy) == 0:
            return 0.0
        
        # 假设最低 10% 的帧是噪声
        sorted_energy = np.sort(frame_energy)
        noise_energy = np.mean(sorted_energy[:max(1, len(sorted_energy) // 10)])
        signal_energy = np.mean(sorted_energy[len(sorted_energy) // 2:])
        
        if noise_energy > 0:
            snr = 10 * np.log10(signal_energy / noise_energy + 1e-10)
            return float(np.clip(snr, 0, 50))
        return 20.0  # 默认值
    
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
