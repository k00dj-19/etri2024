ckpt_path=/home/previ2401/project/QD-DETR-main/results/hl-video_tef-exp-2024_02_14_10_48_47/model_best.ckpt
eval_split_name='val'
t2_feat_dir=/home/previ2401/project/features/clip_paraphrase/
a_feat_type=pann
a_feat_dim=2050
feat_root=../features
a_feat_dir=${feat_root}/pann_features/
eval_path=data/highlight_${eval_split_name}_release.jsonl

PYTHONPATH=$PYTHONPATH:. python qd_detr/inference.py \
--resume ${ckpt_path} \
--eval_split_name ${eval_split_name} \
--eval_path ${eval_path} \
--t2_feat_dir ${t2_feat_dir} \
--t3_feat_dir ${t3_feat_dir} \
--t4_feat_dir ${t4_feat_dir} \
--a_feat_dir ${a_feat_dir} \
--a_feat_dim ${a_feat_dim} \
${@:3}
