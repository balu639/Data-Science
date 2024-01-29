import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import nltk
import torch
from sklearn.metrics import classification_report
import sys

def identify_metaphor_sentence(txt, metaphor):
  for sentence in nltk.sent_tokenize(txt):
    if metaphor in sentence:
      return sentence

nltk.download('punkt')

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

    # Split the data into training and testing sets
    train_data, test_data = train_test_split(df, test_size=0.2, random_state=42)

    # Load pre-trained BERT model and tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)

    # Tokenize and encode the text data, including metaphor word embeddings
    train_tokens = tokenizer(list(train_data["text"]), list(train_data["metaphor_word"]), padding=True, truncation=True, return_tensors="pt")
    test_tokens = tokenizer(list(test_data["text"]), list(test_data["metaphor_word"]), padding=True, truncation=True, return_tensors="pt")


    # Create PyTorch DataLoader
    train_dataset = TensorDataset(
        train_tokens["input_ids"],
        train_tokens["attention_mask"],
        torch.tensor(list(train_data["label_boolean"].astype(int))),
        torch.tensor(list(train_tokens["input_ids"][:, 1]))  # Use the second token as the metaphor word embedding
    )

    test_dataset = TensorDataset(
        test_tokens["input_ids"],
        test_tokens["attention_mask"],
        torch.tensor(list(test_data["label_boolean"].astype(int))),
        torch.tensor(list(test_tokens["input_ids"][:, 1]))  # Use the second token as the metaphor word embedding
    )

    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)
    return train_data, test_data, train_loader, test_loader, model

def train(train_data, test_data, train_loader, test_loader, model):
    # Fine-tune the BERT model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer = AdamW(model.parameters(), lr=5e-5)
    early_stopping_patience = 5
    best_loss = float('inf')
    current_patience = 0

    losses = []
    num_epochs = 10
    for epoch in range(num_epochs):
        model.train()
        for batch in train_loader:
            input_ids, attention_mask, labels, metaphor_ids = batch
            input_ids, attention_mask, labels, metaphor_ids = (
                input_ids.to(device),
                attention_mask.to(device),
                labels.to(device),
                metaphor_ids.to(device),
            )

            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
        losses.append(loss)
        print(f"epoch {epoch} loss {loss}")
        if loss < best_loss:
          best_loss = loss
          current_patience = 0
        else:
          current_patience += 1
          if current_patience >= early_stopping_patience:
            print(f"Early stopping at epoch {epoch + 1} with loss: {best_loss}")
            break

    loss_values = [loss.item() for loss in losses]

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
    predictions_df = predictions_df.iloc[:, [0, 2, 1, 3]]
    return loss_values, model, test_report, predictions_df

# def plot(loss_values):
#     epochs = list(range(1, len(loss_values) + 1))
#     # Plot the loss values over epochs
#     plt.plot(epochs, loss_values, marker='o', linestyle='-')
#     plt.title('Loss Over Epochs')
#     plt.xlabel('Epoch')
#     plt.ylabel('Loss')
#     plt.grid(True)
#     plt.show()


def main():
    # Check if a file name is provided as an argument
    if len(sys.argv) != 2:
        print("Usage: python script.py <file_name>")
        sys.exit(1)

    # Get the file name from the command line argument
    file_name = sys.argv[1]
    data = file_name
    train_data, test_data, train_loader, test_loader, model = load_data_and_model(data)
    loss_values, model, test_report, predictions_df = train(train_data, test_data, train_loader, test_loader, model)
    # plot(loss_values)
    print(test_report)
    print(predictions_df)
    model.save_pretrained('sml_bert_final')


if __name__ == "__main__":
    main()



