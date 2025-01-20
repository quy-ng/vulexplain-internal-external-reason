import asyncio, langchain, os, click, evaluate
from asyncio_throttle import Throttler
from langchain_community.cache import SQLiteCache
from langchain.prompts.chat import ChatPromptTemplate
from langchain.chains import LLMChain
from langchain_community.chat_models import ChatOpenAI
from langchain.schema import BaseOutputParser
from project_dataset import load_dataset
from tqdm.asyncio import tqdm
from dotenv import load_dotenv

load_dotenv()
RATE_LIMIT = 2

rouge = evaluate.load("rouge")
bleu = evaluate.load("bleu")
bertscore = evaluate.load("bertscore")


task_def = {
    'attack_vector': 'delineates the exploitation technique',
    'root_cause': 'highlights the preconditions leading to a vulnerability',
    'impact': 'indicates the potential consequences',
    'vulnerability_type': 'offers an explanation to possible function vulnerabilities'
}

sys_prompt = """
    You are a software engineer in C/C++ and know about vulnerability and cyber-security. 
    Given a code snippet, you will return a line of text, which is
"""
user_template = "{text}"


def get_loc(sample, linevul_ranking, linevul_top=10):
    v = linevul_ranking.tolist()
    breaked_lines = sample.split('\n')
    new_sample = '\n'.join([breaked_lines[i] for i in v[:linevul_top]]).strip()
    return new_sample

def store_results_to_file(results, file_path):
    with open(file_path, 'w') as file:
        for result in results:
            file.write(f'{result[0]}\n')

def _load_dataset(task):
    ds = load_dataset(task)
    train_ds = ds['train'].to_pandas()
    test_ds = ds['test'].to_pandas()
    return train_ds, test_ds


class OutputParser(BaseOutputParser):
    """Parse the output of an LLM call if needed"""
    def parse(self, resp: str):
        return resp


def call_openai_api_(data: list, openai_api: str, task: str):
    throttler = Throttler(rate_limit=RATE_LIMIT)
    langchain.llm_cache = SQLiteCache(database_path=f".prompt_{task}.langchain.db")

    r = asyncio.run(handle_openai(data, openai_api, throttler, task))
    final_result = [i['text'] for i in r]
    return final_result


async def handle_openai(df, openai_key, throttler, task):
    tasks = []
    prompt = sys_prompt + task + '' + task_def[task]

    final_prompt = ChatPromptTemplate.from_messages([
        ("system", prompt),
        ("human", user_template),
    ])
    llm = ChatOpenAI(temperature=0,
                        openai_api_key=openai_key, model='gpt-4o',
                        request_timeout=1200, max_retries=2, max_tokens=4095,
                        model_kwargs={"top_p": 1})
    chain = LLMChain(llm=llm, prompt=final_prompt, output_parser=OutputParser())

    for i, row in df.iterrows():
        tasks += [async_llm_call(chain, row, throttler)]

    results = await tqdm.gather(*tasks)
    return results


async def async_llm_call(chain, data, throttler):
    try:
        async with throttler:
            await asyncio.sleep(3)
            doc = get_loc(data['processed_func'], data['linevul_ranking'] )
            resp = await chain.ainvoke(doc)
        return resp

    except Exception as e:
        raise e

@click.command()
@click.option('--task', help='Task to perform')
def main(task):
    # Load dataset
    train_ds, test_ds = _load_dataset(task)

    # OpenAI API key
    openai_api_key = os.environ.get("OPENAI_KEY")

    # Call OpenAI API
    final_data = call_openai_api_(test_ds, openai_api_key, task)

    label = []
    for i_row, row in test_ds.iterrows():
        target = row["explain"]
        label.append(target)


    rouge_results = []
    bleu_results = []
    bert_results = []

    dump_ = []

    for v in zip(final_data, label):
        dump_.append(v)
        r_ = rouge.compute(predictions=[v[0]], references=[v[1]])
        b_ = bleu.compute(predictions=[v[0]], references=[v[1]])
        bert_ = bertscore.compute(predictions=[v[0]], references=[v[1]], lang="en")
        rouge_results.append((r_['rouge1'], r_['rouge2'], r_['rougeL']))
        bleu_results.append(b_)
        bert_results.append(bert_['f1'][0])
    

    store_results_to_file(dump_, f"{task}_zeroshot.txt")


    rouge_results_array = np.array(rouge_results)
    bleu_results_array = np.array([b['bleu'] for b in bleu_results])
    avg_rouge1 = np.mean(rouge_results_array[:, 0])
    avg_rouge2 = np.mean(rouge_results_array[:, 1])
    avg_rougeL = np.mean(rouge_results_array[:, 2])
    avg_bleu = np.mean(bleu_results_array)
    avg_bert = np.mean(bert_results)

    # print Rouge
    print("Average Rouge-1:", avg_rouge1)
    print("Average Rouge-2:", avg_rouge2)
    print("Average Rouge-L:", avg_rougeL)
    # print BLEU
    print("Average BLEU:", avg_bleu)
    print("Average BERT:", avg_bert)

if __name__ == "__main__":
    main()
