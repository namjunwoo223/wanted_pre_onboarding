from sentence_transformers import util

def cal_cos_score(emb1, emb2):
    return util.cos_sim(emb1, emb2) #Sentence transformers 패키지에 있는 util안에 모델을 통한 코사인 유사도 결과를 리턴해줌.
