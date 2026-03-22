"""
上海话声韵母分析模块

基于 RIME 吴语上海话词库 (wugniu_zaonhe.dict.yaml) 解析声韵母

吴语拉丁式注音法 (法吴) 声母系统:
- 清塞音/塞擦音: p, ph, t, th, k, kh, ts, tsh, c, ch
- 浊塞音/塞擦音: b, d, g, dz, j
- 清擦音: f, s, sh, h
- 浊擦音: v, z, zh, gh
- 鼻音: m, n, ng, gn (ny), nk
- 边音: l
- 零声母: (空)

韵母系统:
- 单元音: a, o, e, i, u, y, oe
- 复元音: au, eu, ou, ae, oe, ia, io, iu, ua, ue, yo
- 鼻韵母: an, en, in, on, un, ang, ong, aon
- 入声韵: aq, eq, iq, oq, uq, yq, oeq

回退策略:
- 如果词库中找不到某个字，使用 pypinyin 获取普通话拼音
- 通过普通话拼音找同音字，再用同音字的上海话读音来近似
"""

import os
import re
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path

# 尝试导入 pypinyin
try:
    import pypinyin
    HAS_PYPINYIN = True
except ImportError:
    HAS_PYPINYIN = False
    print("Warning: pypinyin not installed. Fallback to mandarin homophone will be disabled.")


class ShanghaiPhonemeAnalyzer:
    """
    上海话声韵母分析器
    
    使用 RIME 吴语上海话词库进行汉字到声韵母的映射
    """
    
    # 吴语拉丁式注音法声母列表 (按长度降序排列，确保长声母优先匹配)
    INITIALS = [
        # 三字母声母
        'tsh', 'ngh',
        # 双字母声母
        'ph', 'th', 'kh', 'ts', 'ch', 'dz', 'sh', 'zh', 'gh', 'ng', 'gn', 'nk',
        # 单字母声母
        'p', 'b', 'm', 'f', 'v',
        't', 'd', 'n', 'l',
        'k', 'g', 'h',
        'c', 'j',
        's', 'z',
        'y', 'w',
    ]
    
    # 浊辅音声母 (上海话特色)
    VOICED_INITIALS = {'b', 'd', 'g', 'dz', 'j', 'v', 'z', 'zh', 'gh', 'm', 'n', 'ng', 'gn', 'l'}
    
    # 送气声母
    ASPIRATED_INITIALS = {'ph', 'th', 'kh', 'tsh', 'ch'}
    
    # 鼻音声母
    NASAL_INITIALS = {'m', 'n', 'ng', 'gn', 'nk'}
    
    # 入声韵母 (以 q 结尾)
    CHECKED_FINALS = {'aq', 'eq', 'iq', 'oq', 'uq', 'yq', 'oeq', 'iaq', 'ioq', 'uaq', 'ueq', 'yoq'}
    
    # 鼻韵母
    NASAL_FINALS = {'an', 'en', 'in', 'on', 'un', 'ang', 'ong', 'aon', 'ian', 'ion', 'uan', 'uen', 'yan', 'yon'}
    
    def __init__(self, dict_path: Optional[str] = None, enable_fallback: bool = True):
        """
        初始化分析器
        
        Args:
            dict_path: 词库文件路径，默认使用模块目录下的 wugniu_zaonhe.dict.yaml
            enable_fallback: 是否启用普通话同音字回退
        """
        if dict_path is None:
            dict_path = os.path.join(os.path.dirname(__file__), 'wugniu_zaonhe.dict.yaml')
        
        self.dict_path = dict_path
        self.enable_fallback = enable_fallback and HAS_PYPINYIN
        self.char_to_pinyin: Dict[str, List[str]] = {}  # 汉字 -> 上海话拼音列表
        self.pinyin_to_chars: Dict[str, List[str]] = {}  # 上海话拼音 -> 汉字列表
        self.mandarin_to_chars: Dict[str, List[str]] = {}  # 普通话拼音 -> 汉字列表 (用于回退)
        self.fallback_cache: Dict[str, Optional[Tuple[str, str]]] = {}  # 回退缓存
        
        # 加载词库
        if os.path.exists(dict_path):
            self._load_dict()
            if self.enable_fallback:
                self._build_mandarin_index()
    
    def _load_dict(self):
        """加载 RIME 词库"""
        with open(self.dict_path, 'r', encoding='utf-8') as f:
            in_header = True
            for line in f:
                line = line.strip()
                
                # 跳过头部
                if line == '...':
                    in_header = False
                    continue
                if in_header:
                    continue
                
                # 跳过空行和注释
                if not line or line.startswith('#'):
                    continue
                
                # 解析: 汉字\t拼音[\t权重]
                parts = line.split('\t')
                if len(parts) < 2:
                    continue
                
                char_or_word = parts[0]
                pinyin = parts[1]
                
                # 只处理单字
                if len(char_or_word) == 1:
                    char = char_or_word
                    
                    # 添加到映射
                    if char not in self.char_to_pinyin:
                        self.char_to_pinyin[char] = []
                    if pinyin not in self.char_to_pinyin[char]:
                        self.char_to_pinyin[char].append(pinyin)
                    
                    if pinyin not in self.pinyin_to_chars:
                        self.pinyin_to_chars[pinyin] = []
                    if char not in self.pinyin_to_chars[pinyin]:
                        self.pinyin_to_chars[pinyin].append(char)
        
        print(f"Loaded {len(self.char_to_pinyin)} characters from {self.dict_path}")
    
    def _build_mandarin_index(self):
        """
        构建普通话拼音到汉字的索引
        用于在词库中找不到字时，通过普通话同音字回退
        """
        if not HAS_PYPINYIN:
            return
        
        for char in self.char_to_pinyin.keys():
            try:
                # 获取普通话拼音 (不带声调)
                mandarin_pinyin = pypinyin.lazy_pinyin(char, style=pypinyin.Style.NORMAL)
                if mandarin_pinyin:
                    py = mandarin_pinyin[0].lower()
                    if py not in self.mandarin_to_chars:
                        self.mandarin_to_chars[py] = []
                    if char not in self.mandarin_to_chars[py]:
                        self.mandarin_to_chars[py].append(char)
            except Exception:
                pass
        
        print(f"Built mandarin index with {len(self.mandarin_to_chars)} pinyin entries")
    
    def _fallback_by_mandarin(self, char: str) -> Optional[Tuple[str, str]]:
        """
        通过普通话拼音找同音字，获取上海话读音
        
        策略:
        1. 获取目标字的普通话拼音
        2. 在词库中找同音字
        3. 返回同音字的上海话读音
        
        Args:
            char: 要查找的汉字
            
        Returns:
            (声母, 韵母) 元组，如果找不到返回 None
        """
        if not self.enable_fallback:
            return None
        
        # 检查缓存
        if char in self.fallback_cache:
            return self.fallback_cache[char]
        
        try:
            # 获取普通话拼音
            mandarin_pinyin = pypinyin.lazy_pinyin(char, style=pypinyin.Style.NORMAL)
            if not mandarin_pinyin:
                self.fallback_cache[char] = None
                return None
            
            py = mandarin_pinyin[0].lower()
            
            # 在词库中找同音字
            if py in self.mandarin_to_chars:
                homophones = self.mandarin_to_chars[py]
                # 取第一个同音字的上海话读音
                for homophone in homophones:
                    if homophone in self.char_to_pinyin:
                        shanghai_pinyin = self.char_to_pinyin[homophone][0]
                        result = self.parse_pinyin(shanghai_pinyin)
                        self.fallback_cache[char] = result
                        return result
            
            self.fallback_cache[char] = None
            return None
            
        except Exception:
            self.fallback_cache[char] = None
            return None
    
    def parse_pinyin(self, pinyin: str) -> Tuple[str, str]:
        """
        解析拼音为声母和韵母
        
        Args:
            pinyin: 吴语拉丁式拼音
            
        Returns:
            (声母, 韵母) 元组，零声母时声母为空字符串
        """
        if not pinyin:
            return ('', '')
        
        pinyin = pinyin.lower()
        
        # 尝试匹配声母 (从长到短)
        for initial in self.INITIALS:
            if pinyin.startswith(initial):
                final = pinyin[len(initial):]
                return (initial, final)
        
        # 没有匹配到声母，整个是韵母 (零声母)
        return ('', pinyin)
    
    def get_char_phoneme(self, char: str, use_fallback: bool = True) -> Optional[Tuple[str, str]]:
        """
        获取单个汉字的声韵母
        
        Args:
            char: 单个汉字
            use_fallback: 是否使用普通话同音字回退
            
        Returns:
            (声母, 韵母) 元组，如果找不到返回 None
        """
        # 直接在词库中查找
        if char in self.char_to_pinyin:
            pinyin = self.char_to_pinyin[char][0]
            return self.parse_pinyin(pinyin)
        
        # 词库中找不到，尝试通过普通话同音字回退
        if use_fallback and self.enable_fallback:
            return self._fallback_by_mandarin(char)
        
        return None
    
    def analyze_text(self, text: str) -> Dict[str, Any]:
        """
        分析文本的声韵母分布
        
        Args:
            text: 输入文本
            
        Returns:
            包含声韵母分布统计的字典
        """
        initial_counts: Dict[str, int] = {}
        final_counts: Dict[str, int] = {}
        voiced_count = 0
        aspirated_count = 0
        nasal_initial_count = 0
        checked_count = 0  # 入声
        nasal_final_count = 0
        total_chars = 0
        unknown_chars: List[str] = []
        
        for char in text:
            # 跳过非汉字
            if not '\u4e00' <= char <= '\u9fff':
                continue
            
            phoneme = self.get_char_phoneme(char)
            if phoneme is None:
                unknown_chars.append(char)
                continue
            
            initial, final = phoneme
            total_chars += 1
            
            # 统计声母
            initial_key = initial if initial else '(零声母)'
            initial_counts[initial_key] = initial_counts.get(initial_key, 0) + 1
            
            # 统计韵母
            if final:
                final_counts[final] = final_counts.get(final, 0) + 1
            
            # 统计特征
            if initial in self.VOICED_INITIALS:
                voiced_count += 1
            if initial in self.ASPIRATED_INITIALS:
                aspirated_count += 1
            if initial in self.NASAL_INITIALS:
                nasal_initial_count += 1
            
            # 检查入声和鼻韵母
            if final:
                if final.endswith('q') or final in self.CHECKED_FINALS:
                    checked_count += 1
                if final.endswith('n') or final.endswith('ng') or final in self.NASAL_FINALS:
                    nasal_final_count += 1
        
        # 计算比例
        voiced_ratio = voiced_count / total_chars if total_chars > 0 else 0.0
        aspirated_ratio = aspirated_count / total_chars if total_chars > 0 else 0.0
        nasal_initial_ratio = nasal_initial_count / total_chars if total_chars > 0 else 0.0
        checked_ratio = checked_count / total_chars if total_chars > 0 else 0.0
        nasal_final_ratio = nasal_final_count / total_chars if total_chars > 0 else 0.0
        
        return {
            'total_chars': total_chars,
            'initial_distribution': initial_counts,
            'final_distribution': final_counts,
            'voiced_initial_ratio': voiced_ratio,
            'aspirated_initial_ratio': aspirated_ratio,
            'nasal_initial_ratio': nasal_initial_ratio,
            'checked_final_ratio': checked_ratio,  # 入声比例
            'nasal_final_ratio': nasal_final_ratio,
            'unknown_chars': unknown_chars,
            'coverage': total_chars / (total_chars + len(unknown_chars)) if (total_chars + len(unknown_chars)) > 0 else 0.0,
        }
    
    def get_initial_category(self, initial: str) -> str:
        """
        获取声母的类别
        
        Returns:
            类别名称: voiced (浊音), aspirated (送气), nasal (鼻音), 
                     voiceless (清音), lateral (边音), zero (零声母)
        """
        if not initial:
            return 'zero'
        if initial in self.VOICED_INITIALS:
            if initial in self.NASAL_INITIALS:
                return 'nasal'
            if initial == 'l':
                return 'lateral'
            return 'voiced'
        if initial in self.ASPIRATED_INITIALS:
            return 'aspirated'
        return 'voiceless'
    
    def get_final_category(self, final: str) -> str:
        """
        获取韵母的类别
        
        Returns:
            类别名称: checked (入声), nasal (鼻韵母), open (开口韵)
        """
        if not final:
            return 'unknown'
        if final.endswith('q') or final in self.CHECKED_FINALS:
            return 'checked'
        if final.endswith('n') or final.endswith('ng') or final in self.NASAL_FINALS:
            return 'nasal'
        return 'open'


# 全局实例
_analyzer: Optional[ShanghaiPhonemeAnalyzer] = None


def get_analyzer() -> ShanghaiPhonemeAnalyzer:
    """获取全局分析器实例"""
    global _analyzer
    if _analyzer is None:
        _analyzer = ShanghaiPhonemeAnalyzer()
    return _analyzer


def analyze_shanghai_phonemes(text: str) -> Dict[str, Any]:
    """
    分析文本的上海话声韵母分布
    
    Args:
        text: 输入文本
        
    Returns:
        包含声韵母分布统计的字典
    """
    return get_analyzer().analyze_text(text)


if __name__ == '__main__':
    # 测试
    analyzer = ShanghaiPhonemeAnalyzer()
    
    # 测试单字
    test_chars = ['在', '問', '熱', '儂', '阿', '是', '我', '你', '好', '吃']
    print("\n单字测试:")
    for char in test_chars:
        phoneme = analyzer.get_char_phoneme(char)
        if phoneme:
            initial, final = phoneme
            pinyins = analyzer.char_to_pinyin.get(char, [])
            print(f"  {char}: 声母={initial or '(零)'}, 韵母={final}, 拼音={pinyins}")
        else:
            print(f"  {char}: 未找到")
    
    # 测试文本分析
    test_text = "侬好，阿拉是上海人，欢迎侬来上海白相。"
    print(f"\n文本分析: {test_text}")
    result = analyzer.analyze_text(test_text)
    print(f"  总字数: {result['total_chars']}")
    print(f"  浊音比例: {result['voiced_initial_ratio']:.2%}")
    print(f"  入声比例: {result['checked_final_ratio']:.2%}")
    print(f"  鼻韵母比例: {result['nasal_final_ratio']:.2%}")
    print(f"  覆盖率: {result['coverage']:.2%}")
    print(f"  声母分布: {result['initial_distribution']}")
    print(f"  未识别字: {result['unknown_chars']}")
