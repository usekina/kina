"""Curated general-wellness actions independent of speech measurements."""
from __future__ import annotations

WHO_GUIDELINE = 'https://www.who.int/publications/i/item/9789241550543'
MEDITERRANEAN_TRIAL = 'https://pubmed.ncbi.nlm.nih.gov/23670794/'
SOCIAL_ENGAGEMENT_STUDY = 'https://pmc.ncbi.nlm.nih.gov/articles/PMC6778491/'
NIA_GUIDE = 'https://www.nia.nih.gov/health/cognitive-health-and-older-adults'
ACTION_IDS = ('social_connection', 'physical_activity', 'balanced_meal', 'learning_interest')

COPY = {
 'English': {
  'heading': 'One small action for yourself today',
  'intro': 'Choose something you would like to do. You can write your own idea or skip today.',
  'boundary': 'These general wellness options are separate from your speech results. They do not explain or treat a score change. Choose what fits your abilities, dietary needs and professional advice.',
  'provenance': 'Options are curated from public health guidance. AI has not assessed your health or prescribed an activity. Your choice is saved in your account; it is not sent to an AI service.',
  'choose': 'What would you like to do?', 'custom': 'Choose my own action', 'skip': 'Skip today',
  'custom_label': 'My small action', 'custom_help': 'Optional, up to 160 characters. Avoid names, contact details and medical information.',
  'save': 'Save my choice', 'saved': 'Your choice was saved.', 'choose_first': 'Choose an option first.',
  'custom_required': 'Write a short action or choose another option.',
  'planned': 'Planned', 'completed': 'Completed by you', 'skipped': 'Skipped',
  'mark': 'I did this', 'done': 'Completion recorded. This does not establish an effect on speech scores.',
  'current': 'Today’s choice', 'source_heading': 'Why are these options here?', 'source_link': 'Read the source',
  'evidence': 'The source discusses general healthy-aging habits. It does not show that this activity changes your recording, or establish a personal duration or dose.',
  'social_connection': 'Connect with someone I want to talk to',
  'physical_activity': 'Choose comfortable movement that is safe for me',
  'balanced_meal': 'Add vegetables or legumes to a meal that fits my dietary needs',
  'learning_interest': 'Spend time on an interest or learning something I enjoy',
  'observation': 'About your recent recordings',
  'no_comparison': 'Two comparable recordings are not available yet. You can still choose a small action.',
  'observation_boundary': 'These are feature-index point differences between your latest recording and its previous comparable recording, not percentages or health changes. Topic and recording conditions may affect results.',
 },
 '日本語': {
  'heading': '今日、自分のためにできる小さなこと',
  'intro': 'やってみたいことを一つ選んでください。自分で書くことも、今日は見送ることもできます。',
  'boundary': '一般的なウェルネスの選択肢であり、発話結果とは別です。スコア変化の原因説明や治療ではありません。体調、能力、食事制限、専門家の助言に合うものを選んでください。',
  'provenance': '公的な健康情報を参考にした選択肢です。AIによる健康評価や処方ではありません。選択はアカウントに保存され、AIサービスには送信されません。',
  'choose': '今日は何をしてみたいですか？', 'custom': '自分で行動を決める', 'skip': '今日は見送る',
  'custom_label': '自分で選ぶ小さな行動', 'custom_help': '任意、160文字以内。氏名、連絡先、医療情報は書かないでください。',
  'save': '選択を保存', 'saved': '選択を保存しました。', 'choose_first': '選択肢を選んでください。',
  'custom_required': '短い行動を書くか、別の選択肢を選んでください。',
  'planned': '予定', 'completed': '自分で完了を記録', 'skipped': '見送り',
  'mark': '実行しました', 'done': '完了を記録しました。発話スコアへの効果を示すものではありません。',
  'current': '今日の選択', 'source_heading': 'この選択肢の根拠', 'source_link': '出典を見る',
  'evidence': '出典は健康的な加齢に関する一般的な習慣を説明しています。この行動が今回の録音を変えることや、個人に適した時間・量を示すものではありません。',
  'social_connection': '話したい人と交流する',
  'physical_activity': '安全にできる無理のない身体活動を選ぶ',
  'balanced_meal': '食事制限に合わせ、野菜や豆類を一食に加える',
  'learning_interest': '好きな趣味や学びに時間を使う',
  'observation': '最近の録音について',
  'no_comparison': '比較可能な録音がまだ2回そろっていません。小さな行動は今でも選べます。',
  'observation_boundary': '最新の録音と直前の比較可能な録音の特徴指数の点差です。割合や健康状態の変化ではありません。話題や録音条件にも影響されます。',
 },
 '中文': {
  'heading': '今天，为自己选一件小事',
  'intro': '选择你愿意做的一件事。也可以自己填写，或今天先跳过。',
  'boundary': '这些日常行动与语音结果分开，不解释或治疗分数变化。请选择符合自己身体条件、饮食需要及专业建议的做法。',
  'provenance': '选项参考公共健康指导整理，不是AI对你健康状态的判断或处方。选择会保存在你的账号中，不会发送给AI服务。',
  'choose': '今天，你想为自己做什么？', 'custom': '我想做另一件事', 'skip': '今天先跳过',
  'custom_label': '我的小行动', 'custom_help': '自愿填写，最多160字。请勿填写姓名、联系方式或医疗信息。',
  'save': '保存我的选择', 'saved': '已保存你的选择。', 'choose_first': '请先选择一个选项。',
  'custom_required': '请简短填写行动，或选择其他选项。',
  'planned': '计划做', 'completed': '你已确认完成', 'skipped': '已跳过',
  'mark': '我已经做了', 'done': '已记录完成。这不代表该行动改变了语音分数。',
  'current': '今天的选择', 'source_heading': '为什么提供这些选项？', 'source_link': '查看原始来源',
  'evidence': '来源讨论一般健康老龄化习惯，不能证明这个行动会改变你的本次录音，也没有为你个人确定时长或剂量。',
  'social_connection': '联系一位我想聊聊的人',
  'physical_activity': '选择一次适合自己、能够安全完成的身体活动',
  'balanced_meal': '在符合饮食需要的前提下，为一餐增加蔬菜或豆类',
  'learning_interest': '做一点喜欢的兴趣活动，或学一点新东西',
  'observation': '关于你的近期记录',
  'no_comparison': '目前还没有两次可比较的记录。你仍然可以选择今天的小行动。',
  'observation_boundary': '下面是最新记录与此前一次可比较记录的特征指数点数差，不是百分比，也不是健康变化。话题和录音条件都可能影响结果。',
 },
}


def wellness_suggestions(language: str, scores: list[dict]) -> dict:
    del scores  # Never rank lifestyle actions by speech scores.
    copy = COPY.get(language, COPY['English'])
    return {
        **copy,
        'habit_labels': {key: copy[key] for key in ACTION_IDS},
        'suggestions': [{'id': key, 'text': copy[key], 'source': NIA_GUIDE,
                         'source_title': 'NIA · Cognitive Health and Older Adults',
                         'evidence': copy['evidence']} for key in ACTION_IDS],
    }
