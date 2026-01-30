import pandas as pd
import numpy as np
import re
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, precision_score, recall_score
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pickle
class TagClassifier(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(TagClassifier, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, output_dim),
            nn.Sigmoid()
        )
    def forward(self, x):
        return self.network(x)
    

class FastTagPredictor:
    def __init__(self):
        print("Начинаем")
        self.embedder = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
        self.mlb = MultiLabelBinarizer()
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Устройство: {self.device}")
        
    def parse_tags(self, tag_string):
        if tag_string is None or (isinstance(tag_string, float) and np.isnan(tag_string)):
            return []
        if not tag_string or str(tag_string).strip() == '':
            return []
        tags = re.sub(r'[\[\]]', '', str(tag_string))
        return [t.strip() for t in tags.split(',') if t.strip()]
    
    def train(self, df, content_col='clean_content', tags_col='clean_tags', 
              epochs=10, batch_size=32, lr=0.001):
        print("Подготовка данных...")
        texts = df[content_col].fillna('').astype(str).tolist()
        tags_list = df[tags_col].apply(self.parse_tags).tolist()
        
        data = [(t, tags) for t, tags in zip(texts, tags_list) if t and tags]
        texts, tags_list = zip(*data)
        
        print("Создание эмбеддингов...")
        X = self.embedder.encode(list(texts), show_progress_bar=True)
        
        print("Подготовка меток...")
        y = self.mlb.fit_transform(tags_list)
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        X_train = torch.FloatTensor(X_train).to(self.device)
        y_train = torch.FloatTensor(y_train).to(self.device)
        X_test = torch.FloatTensor(X_test).to(self.device)
        y_test = torch.FloatTensor(y_test).to(self.device)
        
        train_dataset = TensorDataset(X_train, y_train)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        self.model = TagClassifier(X.shape[1], y.shape[1]).to(self.device)
        
        criterion = nn.BCELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=lr)

        print("Начинаем обучение...")
        self.model.train()
        for epoch in range(epochs):
            total_loss = 0
            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            
            self.model.eval()
            with torch.no_grad():
                val_outputs = self.model(X_test)
                val_loss = criterion(val_outputs, y_test)
                val_pred = (val_outputs > 0.05).float()
                accuracy = (val_pred == y_test).float().mean()
            
            
            print(f"Эпоха {epoch+1}/{epochs} - Loss: {total_loss/len(train_loader):.4f}, "
                  f"Val Loss: {val_loss:.4f}, Val Acc: {accuracy:.4f}")
            self.model.train()
        
        self.model.eval()
        with torch.no_grad():
            test_pred = (self.model(X_test) > 0.3).float()
            final_accuracy = (test_pred == y_test).float().mean()
            print(f"\nФинальная точность на тесте: {final_accuracy:.4f}")
            
    def predict(self, text, threshold=0.11, boost_factor=3.0):
        self.model.eval()
        X = self.embedder.encode([text])
        X = torch.FloatTensor(X).to(self.device)
        
        with torch.no_grad():
            probs = self.model(X)[0].cpu().numpy()
        
        clean_text = re.sub(r'[^\w\s]', ' ', text.lower())
        text_words = set(clean_text.split())
        
        final_probs = probs.copy()
        
        for i, tag_name in enumerate(self.mlb.classes_):
            clean_tag = str(tag_name).strip("'\" ").lower()
            
            tag_root = clean_tag[:4] if len(clean_tag) > 4 else clean_tag
            
            found = False
            if clean_tag in clean_text: 
                found = True
            else: 
                for word in text_words:
                    if word.startswith(tag_root):
                        found = True
                        break
            
            if found:
                final_probs[i] = (final_probs[i] * boost_factor) + 0.05
        
        final_probs = np.clip(final_probs, 0, 1.0)
        
        indices = np.where(final_probs > threshold)[0]
        tags = [(self.mlb.classes_[i], float(final_probs[i])) for i in indices]
        
        return sorted(tags, key=lambda x: x[1], reverse=True)


    def save(self, path='pytorch_tag_model'):
        torch.save(self.model.state_dict(), f'{path}.pth')
        with open(f'{path}.pkl', 'wb') as f:
            pickle.dump({
                'mlb': self.mlb,
                'input_dim': self.model.network[0].in_features,
                'output_dim': self.model.network[-2].out_features
            }, f)
        print(f"Модель сохранена: {path}.pth")
    
    def load(self, path='pytorch_tag_model'):
        with open(f'{path}.pkl', 'rb') as f:
            data = pickle.load(f)
            self.mlb = data['mlb']
            input_dim = data['input_dim']
            output_dim = data['output_dim']
        
        self.model = TagClassifier(input_dim, output_dim).to(self.device)
        self.model.load_state_dict(torch.load(f'{path}.pth', map_location=self.device))
        self.model.eval()
        print(f"Модель загружена: {path}.pth")
