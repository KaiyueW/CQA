import torch

class SalFormer(torch.nn.Module):
    def __init__(self, vision_encoder, bert):
        """
        In the constructor we instantiate four parameters and assign them as
        member parameters.
        """
        super().__init__()

        self.vit = vision_encoder
        self.feature_dim = 768 #each token is in 768
        self.bert = bert

        self.vision_head = torch.nn.Sequential( #linear+relu
            torch.nn.Linear(self.feature_dim, self.feature_dim),
            torch.nn.GELU(),
            torch.nn.Linear(self.feature_dim, self.feature_dim),
            torch.nn.GELU()
        )

        self.text_head = torch.nn.Sequential(
            torch.nn.Linear(self.feature_dim, self.feature_dim),
            torch.nn.GELU(),
            torch.nn.Linear(self.feature_dim, self.feature_dim),
            torch.nn.GELU()
        )

        self.cross_attention = torch.nn.MultiheadAttention(self.feature_dim, 16, kdim=self.feature_dim, vdim=self.feature_dim, batch_first=True)
        self.cross_attention1 = torch.nn.MultiheadAttention(self.feature_dim, 16, kdim=self.feature_dim, vdim=self.feature_dim, batch_first=True)

        self.ln1 = torch.nn.LayerNorm(self.feature_dim)
        self.ln2 = torch.nn.LayerNorm(self.feature_dim)

        self.self_attetion = torch.nn.MultiheadAttention(self.feature_dim, 16, batch_first=True)

        self.text_feature_query = torch.nn.Parameter(torch.randn(10, self.feature_dim).unsqueeze(0)/2) #10 learnable text querys
        self.img_positional_embedding = torch.nn.Parameter(torch.zeros(49, self.feature_dim))
        self.text_positional_embedding = torch.nn.Parameter(torch.zeros(10, self.feature_dim))


        self.dense1 = torch.nn.Linear(self.feature_dim, self.feature_dim)
        self.relu1 = torch.nn.ReLU()

        self.decoder = torch.nn.Sequential(
            torch.nn.Conv2d(self.feature_dim, 512, 3),
            torch.nn.BatchNorm2d(512),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=0.1),
            torch.nn.Conv2d(512, 512, 3),
            torch.nn.BatchNorm2d(512),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=0.1),
            torch.nn.Upsample((16,16), mode='bilinear'),
            torch.nn.Conv2d(512, 256, 3),
            torch.nn.BatchNorm2d(256),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=0.1),
            torch.nn.Conv2d(256, 256, 3),
            torch.nn.BatchNorm2d(256),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=0.1),
            torch.nn.Upsample((32,32), mode='bilinear'),
            torch.nn.Conv2d(256, 128, 3),
            torch.nn.BatchNorm2d(128),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=0.1),
            torch.nn.Conv2d(128, 128, 3),
            torch.nn.BatchNorm2d(128),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=0.1),
            torch.nn.Upsample((130,130), mode='bilinear'),
            torch.nn.Conv2d(128, 1, 3),
            torch.nn.BatchNorm2d(1),
            torch.nn.Sigmoid(),
        )

        self.vit.eval()
        self.bert.eval()
        self.train(True)

    # def eval(self):
    #     super().eval()
    #     self.vit.eval()
    #     self.bert.eval()

    # def train(self, mode=True):
    #     super().train(mode)
    #     self.vit.train(mode)
    #     self.bert.train(mode)

    def forward(self, img, q_inputs):

        img_features =  self.vit.forward(img, return_dict =True)["last_hidden_state"] # 最终输出, [B, 50, 768] 768 is the feature dimension for each token, 50 tokens (49 image tokens + CLS token)
        with torch.no_grad():
            text_features =  self.bert(**q_inputs)["last_hidden_state"] #size is [Batch, max_seq_len, 768]
            # text_features =  torch.unsqueeze(bert_output["last_hidden_state"][:,0,:], 1)
        text_features = self.cross_attention.forward(self.text_feature_query.repeat([text_features.shape[0], 1, 1]), text_features, text_features, need_weights=False)[0]
        # cross(Q, K, V), output is attention_weights：注意力分配表, 某个词对其他词的attn weights; attention_output: 加权融合后的新表示
        # [B, 10, 768]

        fused_features = torch.concat((self.vision_head(img_features)+self.img_positional_embedding, self.text_head(text_features)+self.text_positional_embedding), 1) # fused into [Batch, 59, 768]
        att_fused_features = self.self_attetion.forward(fused_features, fused_features, fused_features, need_weights=False)[0]
        fused_features = fused_features + att_fused_features
        fused_features = self.ln1(fused_features)

        features = self.cross_attention1.forward(img_features, fused_features, fused_features, need_weights=False)[0]
        features = img_features + features
        features = self.ln2(features)

        features = self.dense1(features)
        latent_features = self.relu1(features)

        latent_features = latent_features.permute(0,2,1)
        out = torch.reshape(latent_features, (features.shape[0], self.feature_dim, 7, 7)) # 7*7*768
        out = self.decoder(out) #[B, 1, 128, 128]

        return out
