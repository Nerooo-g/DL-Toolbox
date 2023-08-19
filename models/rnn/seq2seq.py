import torch
from torch import nn
import torch.nn.functional as F

MAX_LENGTH = 200


class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, layers, dropout_p=0.1):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(input_size, hidden_size)
        self.lstm = nn.LSTM(input_size=hidden_size, hidden_size=hidden_size, batch_first=True, num_layers=layers)
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, input):
        embedded = self.dropout(self.embedding(input))
        output, hidden = self.lstm(embedded)
        return output, hidden


def dot_luong_forward(query, keys):
    """

    @param query: batch_size,1,dim
    @param keys: batch_size,seq_len,dim
    @return: batch_size,1,seq_len
    """
    attn_energies = torch.bmm(query, keys.transpose(1, 2))

    return F.softmax(attn_energies, dim=-1)


class AttnDecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, layers, dropout_p=0.1):
        super(AttnDecoderRNN, self).__init__()
        self.embedding = nn.Embedding(output_size, hidden_size)
        self.lstm = nn.LSTM(input_size=hidden_size, hidden_size=hidden_size, batch_first=True, num_layers=layers)
        self.out = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, encoder_outputs, encoder_hidden, target_tensor=None):
        batch_size = encoder_outputs.size(0)
        decoder_input = torch.empty(batch_size, 1, dtype=torch.long, device='cuda:0').fill_(1)
        decoder_hidden = encoder_hidden
        decoder_outputs = []
        attentions = []

        for i in range(MAX_LENGTH):
            decoder_output, decoder_hidden, attn_weights = self.forward_step(
                decoder_input, decoder_hidden, encoder_outputs
            )
            decoder_outputs.append(decoder_output)
            attentions.append(attn_weights)

            if target_tensor is not None:
                # Teacher forcing: Feed the target as the next input
                decoder_input = target_tensor[:, i].unsqueeze(1)  # Teacher forcing
            else:
                # Without teacher forcing: use its own predictions as the next input
                _, topi = decoder_output.topk(1)
                decoder_input = topi.squeeze(-1).detach()  # detach from history as input

        decoder_outputs = torch.cat(decoder_outputs, dim=1)
        decoder_outputs = F.log_softmax(decoder_outputs, dim=-1)
        attentions = torch.cat(attentions, dim=1)

        return decoder_outputs, decoder_hidden, attentions

    def forward_step(self, decoder_input, hidden, encoder_outputs):
        embedded = self.dropout(self.embedding(decoder_input))
        output, decoder_hidden = self.lstm(embedded, hidden)
        attn_energies = dot_luong_forward(output, encoder_outputs)
        output = torch.bmm(attn_energies, encoder_outputs)
        output = self.out(output)

        return output, decoder_hidden, attn_energies

# a = torch.randint(0, 500, (32, 100)).cuda()
# label = torch.randint(0,400,(32,200)).cuda()
# encoder = EncoderRNN(500, 512, 1).cuda()
# decoder = AttnDecoderRNN(512, 400, 1).cuda()
# output, hidden = encoder(a)
# decoder_outputs, decoder_hidden, attentions=decoder(output, hidden,label)
# print(decoder_outputs.shape,decoder_hidden[0].shape,decoder_hidden[1].shape,attentions.shape)
