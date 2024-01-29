import pandas as pd
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from torch.utils.data import DataLoader, TensorDataset
import nltk
import torch
from sklearn.metrics import classification_report
import sys

def identify_metaphor_sentence(txt, metaphor):
  for sentence in nltk.sent_tokenize(txt):
    if metaphor in sentence:
      return sentence
def load_data_and_model(data):
    df = pd.read_csv(data)

    # Drop rows with missing values
    df = df.dropna(subset=['text', 'label_boolean'])

    #replace metophorids with words
    metaphor = {0:'road', 1:'candle', 2:'light', 3:'spice', 4:'ride', 5:'train', 6:'boat'}
    df.replace({"metaphorID": metaphor},inplace=True)
    # replace the text with the first sentence which contains the metaphor word
    df['text'] = df.apply(lambda x: identify_metaphor_sentence(x['text'], x['metaphorID']),axis=1)
    df = df.rename(columns={'metaphorID': 'metaphor_word'})

    #Drop rows with missing values
    df = df.dropna(subset=['text', 'label_boolean'])
    test_data = df
    # Load pre-trained BERT model and tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertForSequenceClassification.from_pretrained('sml_bert_final')

    test_tokens = tokenizer(list(test_data["text"]), list(test_data["metaphor_word"]), padding=True, truncation=True, return_tensors="pt")

    test_dataset = TensorDataset(
        test_tokens["input_ids"],
        test_tokens["attention_mask"],
        torch.tensor(list(test_data["label_boolean"].astype(int))),
        torch.tensor(list(test_tokens["input_ids"][:, 1]))  # Use the second token as the metaphor word embedding
    )

    test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)
    return test_data, test_loader, model


def test(test_data, test_loader, model):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    # Evaluate the fine-tuned model
    model.eval()
    predictions = []
    true_labels = []

    with torch.no_grad():
        for batch in test_loader:
            input_ids, attention_mask, labels, metaphor_ids = batch
            input_ids, attention_mask, labels, metaphor_ids = (
                input_ids.to(device),
                attention_mask.to(device),
                labels.to(device),
                metaphor_ids.to(device),
            )

            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            predictions.extend(torch.argmax(logits, dim=1).cpu().numpy())
            true_labels.extend(labels.cpu().numpy())

    test_report = classification_report(true_labels, predictions)
    predictions_df = test_data
    predictions_df.insert(3, 'Predictions', predictions)
    predictions_df.iloc[:, 3] = predictions_df.iloc[:, 3].astype(bool)
    predictions_df = predictions_df.iloc[:,[0,2,1,3]]
    return test_report, predictions_df

def main():
    # Check if a file name is provided as an argument
    if len(sys.argv) != 2:
        print("Usage: python script.py <file_name>")
        sys.exit(1)

    # Get the file name from the command line argument
    file_name = sys.argv[1]
    data = file_name
    test_data,  test_loader, model = load_data_and_model(data)
    test_report, predictions_df = test(test_data, test_loader, model)
    print(test_report)
    print(predictions_df)

if __name__ == "__main__":
    main()
