from datasets import load_dataset
ds = load_dataset('hyperpartisan_news_detection', 'bypublisher', split='train')
with open('ds_out_utf8.txt', 'w', encoding='utf-8') as f:
    f.write("Features: " + str(ds.features) + "\n")
    f.write("First row: " + str(ds[0]) + "\n")
