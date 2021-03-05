import argparse

def parser():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--bert_model", default="bert-large-cased-whole-word-masking", type=str,
                        help="Bert pre-trained model selected in the list: bert-base-uncased, "
                             "bert-large-uncased, bert-base-cased, bert-base-multilingual, bert-base-chinese.")
    parser.add_argument("--output_dir", default="output_path", type=str,
                        help="The output directory where the model checkpoints and predictions will be written.")

    ## Other parameters
    parser.add_argument("--train_file", default="data/squad/train-v2.0.json",
                        type=str, help="SQuAD json for training. E.g., train-v1.1.json")
    parser.add_argument("--predict_file", default="data/squad/dev-v2.0.json",
                        type=str,
                        help="SQuAD json for predictions. E.g., dev-v1.1.json or test-v1.1.json")

    parser.add_argument("--test_file", default="data/newsqa/test.json",
                        type=str,
                        help="SQuAD json for predictions. E.g., dev-v1.1.json or test-v1.1.json")
    # parser.add_argument("--train_tag_file", default="data/squad/squad_train_span_sample.json", type=str)
    # parser.add_argument("--predict_tag_file", default="data/squad/squad_dev_span_sample.json", type=str)

    parser.add_argument("--squad_data_dir", default="data/squad/", type=str)

    parser.add_argument("--newsqa_data_dir", default="data/newsqa/", type=str)

    parser.add_argument("--max_seq_length", default=288, type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. Sequences "
                             "longer than this will be truncated, and sequences shorter than this will be padded.")
    parser.add_argument("--max_ent_num", default=50, type=int,
                        help="The maximum total refined entity in each passage.")
    parser.add_argument("--doc_stride", default=96, type=int,
                        help="When splitting up a long document into chunks, how much stride to take between chunks.")
    parser.add_argument("--max_query_length", default=48, type=int,
                        help="The maximum number of tokens for the question. Questions longer than this will "
                             "be truncated to this length.")
    parser.add_argument("--path_length", default=3, type=int,
                        help="The maximum length of entity or path for each token pairs.")
    parser.add_argument("--max_token_size", default=3, type=int,
                        help="The maximum number of tokens in each entity or relation.")
    parser.add_argument("--num_inference_layer", default=1, type=int,
                        help="The maximum number of tokens in each entity or relation.")




    parser.add_argument("--do_train", default=True, action='store_true', help="Whether to run training.")
    parser.add_argument("--do_predict", default=False, action='store_true', help="Whether to run eval on the dev set.")
    parser.add_argument("--do_test", default=False, action='store_true', help="Whether to run eval on the test set. (only for NewsQA dataset)")
    parser.add_argument("--train_batch_size", default=4, type=int, help="Total batch size for training.")
    parser.add_argument("--predict_batch_size", default=8, type=int, help="Total batch size for predictions.")
    parser.add_argument("--learning_rate", default=3e-5, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--num_train_epochs", default=3.0, type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--warmup_proportion", default=0.1, type=float,
                        help="Proportion of training to perform linear learning rate warmup for. E.g., 0.1 = 10% "
                             "of training.")
    parser.add_argument("--n_best_size", default=20, type=int,
                        help="The total number of n-best predictions to generate in the nbest_predictions.json "
                             "output file.")
    parser.add_argument("--max_answer_length", default=30, type=int,
                        help="The maximum length of an answer that can be generated. This is needed because the start "
                             "and end predictions are not conditioned on one another.")
    parser.add_argument("--verbose_logging", default=False, action='store_true',
                        help="If true, all of the warnings related to data processing will be printed. "
                             "A number of warnings are expected for a normal SQuAD evaluation.")
    parser.add_argument("--no_cuda",
                        default=False,
                        action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help="random seed for initialization")
    parser.add_argument('--gradient_accumulation_steps',
                        type=int,
                        default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--do_lower_case",
                        action='store_true',
                        help="Whether to lower case the input text. True for uncased models, False for cased models.")
    parser.add_argument("--local_rank",
                        type=int,
                        default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument('--fp16',
                        default=False,
                        action='store_true',
                        help="Whether to use 16-bit float precision instead of 32-bit")
    parser.add_argument('--loss_scale',
                        type=float, default=0,
                        help="Loss scaling to improve fp16 numeric stability. Only used when fp16 set to True.\n"
                             "0 (default value): dynamic loss scaling.\n"
                             "Positive power of 2: static loss scaling value.\n")
    parser.add_argument('--null_score_diff_threshold',
                        type=float, default=0.0,
                        help="If null_score - best_non_null is greater than the threshold predict null.")

    args = parser.parse_args()
    return args

args = parser()