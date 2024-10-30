# Copyright (c) 2024 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from models.tts.maskgct.maskgct_utils import *
from huggingface_hub import hf_hub_download
import safetensors
import soundfile as sf
import jieba

if __name__ == "__main__":
   
    # build model
    device = torch.device("cuda:0")
    cfg_path = "./models/tts/maskgct/config/maskgct.json"
    cfg = load_config(cfg_path)
    # # 1. build semantic model (w2v-bert-2.0)
    # semantic_model, semantic_mean, semantic_std = build_semantic_model(device)
    # # 2. build semantic codec
    # semantic_codec = build_semantic_codec(cfg.model.semantic_codec, device)
    # # 3. build acoustic codec
    # codec_encoder, codec_decoder = build_acoustic_codec(
    #     cfg.model.acoustic_codec, device
    # )
    # # 4. build t2s model
    # t2s_model = build_t2s_model(cfg.model.t2s_model, device)
    # # 5. build s2a model
    # s2a_model_1layer = build_s2a_model(cfg.model.s2a_model.s2a_1layer, device)
    # s2a_model_full = build_s2a_model(cfg.model.s2a_model.s2a_full, device)

    # 1. build semantic model (w2v-bert-2.0)
    semantic_model_data = check_and_load_model('semantic_model.pth', build_semantic_model, device)
    semantic_model, semantic_mean, semantic_std = semantic_model_data

    # 2. build semantic codec
    semantic_codec = check_and_load_model('semantic_codec.pth', build_semantic_codec, cfg.model.semantic_codec, device)

    # 3. build acoustic codec
    codec_data = check_and_load_model('acoustic_codec.pth', build_acoustic_codec, cfg.model.acoustic_codec, device)
    codec_encoder, codec_decoder = codec_data

    # 4. build t2s model
    t2s_model = check_and_load_model('t2s_model.pth', build_t2s_model, cfg.model.t2s_model, device)

    # 5. build s2a model
    s2a_model_1layer = check_and_load_model('s2a_model_1layer.pth', build_s2a_model, cfg.model.s2a_model.s2a_1layer, device)
    s2a_model_full = check_and_load_model('s2a_model_full.pth', build_s2a_model, cfg.model.s2a_model.s2a_full, device)

    # download checkpoint
    # download semantic codec ckpt
    # semantic_code_ckpt = hf_hub_download(
    #     "amphion/MaskGCT", filename="semantic_codec/model.safetensors"
    # )
    # # download acoustic codec ckpt
    # codec_encoder_ckpt = hf_hub_download(
    #     "amphion/MaskGCT", filename="acoustic_codec/model.safetensors"
    # )
    # codec_decoder_ckpt = hf_hub_download(
    #     "amphion/MaskGCT", filename="acoustic_codec/model_1.safetensors"
    # )
    # # download t2s model ckpt
    # t2s_model_ckpt = hf_hub_download(
    #     "amphion/MaskGCT", filename="t2s_model/model.safetensors"
    # )
    # # download s2a model ckpt
    # s2a_1layer_ckpt = hf_hub_download(
    #     "amphion/MaskGCT", filename="s2a_model/s2a_model_1layer/model.safetensors"
    # )
    # s2a_full_ckpt = hf_hub_download(
    #     "amphion/MaskGCT", filename="s2a_model/s2a_model_full/model.safetensors"
    # )

    # load semantic codec
    safetensors.torch.load_model(semantic_codec, "./models/tts/maskgct/ckpt/semantic_codec/model.safetensors")
    # load acoustic codec
    safetensors.torch.load_model(codec_encoder, "./models/tts/maskgct/ckpt/acoustic_codec/model.safetensors")
    safetensors.torch.load_model(codec_decoder, "./models/tts/maskgct/ckpt/acoustic_codec/model_1.safetensors")
    # load t2s model
    safetensors.torch.load_model(t2s_model, "./models/tts/maskgct/ckpt/t2s_model/model.safetensors")
    # load s2a model
    safetensors.torch.load_model(s2a_model_1layer, "./models/tts/maskgct/ckpt/s2a_model/s2a_model_1layer/model.safetensors")
    safetensors.torch.load_model(s2a_model_full, "./models/tts/maskgct/ckpt/s2a_model/s2a_model_full/model.safetensors")

    # inference
    prompt_wav_path = "./models/tts/maskgct/wav/Will_Lyman.wav"
    save_path = "./myprod_en.wav"
    # prompt_text = " We do not break. We never give in. We never back down."
    prompt_text = "51% of American men say the TV remote has increased their quality of life, especially when it comes to lending. We never bait and switch and we never put you in a situation you'll regret later. In fact, we'll work with you to get the best rate possible. Pretty unusual, huh? The truth, from a financial institution?"
    # prompt_text = "本月早些时候，他抨击了负责监管SpaceX发射的互联网卫星的联邦通信委员会。他在X.com上表示，如果委员会没有“非法撤销”该公司寻求的价值超过8.86亿美元、用以向乡村地区提供互联网接入的联邦资金，那么相关卫星套件“可能会挽救北卡罗来纳州人们的生命”，此前飓风摧毁了该州部分地区。"
    target_text = "Edward Joseph Herlihy (August 14, 1909 – January 30, 1999) was an American newsreel narrator for Universal-International. He was also a long-time radio and television announcer for NBC, hosting The Horn and Hardart Children's Hour in the 1940s and 1950."
    # Specify the target duration (in seconds). If target_len = None, we use a simple rule to predict the target duration.
    # count the number of Chinese words in target_text using jieba package
    # target_len = len(jieba.lcut(target_text)) * 0.5 + 10
    prompt_len = len(prompt_text.split())
    # conv_ratio = None
    target_len = None
    maskgct_inference_pipeline = MaskGCT_Inference_Pipeline(
        semantic_model,
        semantic_codec,
        codec_encoder,
        codec_decoder,
        t2s_model,
        s2a_model_1layer,
        s2a_model_full,
        semantic_mean,
        semantic_std,
        device,
    )

    recovered_audio = maskgct_inference_pipeline.maskgct_inference(
        prompt_wav_path, prompt_text, target_text, "en", "en", target_len=target_len
    )

    sf.write(save_path, recovered_audio, 24000)
