import os
import shutil
import glob
import tensorflow as tf
from object_detection import model_hparams
from object_detection import model_lib

BASE_MODEL_DIR = "/home/baddad/workspace/object-dash/data/models"


class MissingFile(Exception):
    pass


def ensure_files_exist(directory, required_globs):
    for file_glob in required_globs:
        abs_glob = os.path.join(directory, file_glob)
        if not glob.glob(abs_glob):
            raise MissingFile(file_glob)


def render_pipeline_config(model_dir, train_record, eval_record, label_pbtxt, batch_size=2):
    with open(os.path.join(model_dir, 'pipeline.config'), 'r') as fd:
        config = fd.read()

    context = {
        "$BATCH_SIZE": batch_size,
        "$FINE_TUNE_CHECKPOINT": os.path.join(model_dir, 'model.ckpt'),
        "$LABEL_MAP": label_pbtxt,
        "$EVAL_TFRECORD": eval_record,
        "$TRAIN_TFRECORD": train_record,
    }

    for key, value in context.items():
        config = config.replace(key, str(value))

    with open(os.path.join(model_dir, 'pipeline.config'), 'w') as fd:
        fd.write(config)


def create_trainable_model(src_model_name, dest_model_name, train_record, eval_record, label_pbtxt):
    model_dir = os.path.join(BASE_MODEL_DIR, src_model_name)
    dest_model_dir = os.path.join(BASE_MODEL_DIR, dest_model_name)

    ensure_files_exist(model_dir, (
        "model.ckpt.data-*",
        "model.ckpt.index",
        "model.ckpt.meta",
        "pipeline.config"
    ))

    shutil.copytree(model_dir, dest_model_dir)

    checkpoint_path = os.path.join(model_dir, 'checkpoint')
    if os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)

    render_pipeline_config(dest_model_dir, train_record, eval_record, label_pbtxt)


def train_model(model_name, num_train_steps=50000, num_eval_steps=2000):
    model_dir = os.path.join(BASE_MODEL_DIR, model_name)
    hparams_overrides = ""
    pipeline_config_path = os.path.join(model_dir, 'pipeline.config')
    eval_training_data = True

    config = tf.estimator.RunConfig(model_dir=model_dir)

    train_and_eval_dict = model_lib.create_estimator_and_inputs(
        run_config=config,
        hparams=model_hparams.create_hparams(hparams_overrides),
        pipeline_config_path=pipeline_config_path,
        train_steps=num_train_steps,
        eval_steps=num_eval_steps)

    estimator = train_and_eval_dict['estimator']
    train_input_fn = train_and_eval_dict['train_input_fn']
    eval_input_fn = train_and_eval_dict['eval_input_fn']
    eval_on_train_input_fn = train_and_eval_dict['eval_on_train_input_fn']
    predict_input_fn = train_and_eval_dict['predict_input_fn']
    train_steps = train_and_eval_dict['train_steps']
    eval_steps = train_and_eval_dict['eval_steps']

    train_spec, eval_specs = model_lib.create_train_and_eval_specs(
        train_input_fn,
        eval_input_fn,
        eval_on_train_input_fn,
        predict_input_fn,
        train_steps,
        eval_steps,
        eval_on_train_data=eval_training_data)

    tf.estimator.train_and_evaluate(estimator, train_spec, eval_specs[0])


if __name__ == "__main__":
    # create_trainable_model(
    #     'ssd_resnet50_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03',
    #     'tylers_model',
    #     '/home/baddad/workspace/object-dash/data/train.tfrecord',
    #     '/home/baddad/workspace/object-dash/data/eval.tfrecord',
    #     '/home/baddad/workspace/object-dash/data/labels.pbtxt'
    # )

    train_model('tylers_model')
