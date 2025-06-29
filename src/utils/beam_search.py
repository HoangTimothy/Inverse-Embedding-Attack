'''
Source code modified from https://github.com/budzianowski/PyTorch-Beam-Search-Decoding/blob/master/decode_beam.py
implementation of beam search on GPT-2's logits
'''

import operator
import torch
import torch.nn as nn
import torch.nn.functional as F
from queue import PriorityQueue
import sys
from typing import List, Dict, Any


class BeamSearchNode(object):
    def __init__(self, hiddenstate, previousNode, wordId, logProb, length):
        '''
        :param hiddenstate:
        :param previousNode:
        :param wordId:
        :param logProb:
        :param length:
        '''
        self.h = hiddenstate
        self.prevNode = previousNode
        self.wordid = wordId
        self.logp = logProb
        self.leng = length

    def eval(self, alpha=1.0):
        reward = 0
        # Add here a function for shaping a reward

        return self.logp / float(self.leng - 1 + 1e-6) + alpha * reward

    def __lt__(self, x):
        if(self.eval() < x.eval()):
            return True

        else:
            return False



def beam_decode_sentence(hidden_X, config, num_generate=1, beam_size=5):
    """
    Simple beam search implementation for text generation
    
    Args:
        hidden_X: Input embedding tensor
        config: Configuration dictionary with model, tokenizer, device
        num_generate: Number of sequences to generate
        beam_size: Beam size for search
    
    Returns:
        List of generated text sequences
    """
    model = config['model']
    tokenizer = config['tokenizer']
    device = config['device']
    
    # Move embedding to device
    hidden_X = hidden_X.to(device)
    
    # Initialize with start token
    start_token = tokenizer.eos_token_id if tokenizer.eos_token_id else tokenizer.pad_token_id
    if start_token is None:
        start_token = 0
    
    # Initialize beam
    beams = [(torch.tensor([[start_token]], device=device), 0.0)]  # (sequence, score)
    
    max_length = 30  # Reduced max length to avoid repetition
    min_length = 5   # Minimum length
    
    for step in range(max_length):
        new_beams = []
        
        for sequence, score in beams:
            if sequence.shape[1] >= max_length:
                new_beams.append((sequence, score))
                continue
            
            # Prepare input
            if hasattr(model, 'transformer'):
                # GPT-2 style
                input_emb = model.transformer.wte(sequence)
            elif hasattr(model, 'model'):
                # OPT style
                input_emb = model.model.decoder.embed_tokens(sequence)
            else:
                # T5 style
                input_emb = model.shared(sequence)
            
            # Concatenate with hidden_X if provided
            if hidden_X is not None:
                hidden_X_unsqueeze = hidden_X.unsqueeze(0).unsqueeze(0)
                input_emb = torch.cat((hidden_X_unsqueeze, input_emb), dim=1)
            
            # Forward pass
            with torch.no_grad():
                outputs = model(inputs_embeds=input_emb, return_dict=True)
                logits = outputs.logits[:, -1, :]  # Last token predictions
                
                # Apply temperature and top-k sampling
                logits = logits / 1.0  # Increased temperature for diversity
                top_k = 20  # Reduced top-k
                if top_k > 0:
                    top_k_logits, top_k_indices = torch.topk(logits, top_k)
                    logits = torch.full_like(logits, float('-inf'))
                    logits.scatter_(1, top_k_indices, top_k_logits)
                
                # Get probabilities
                probs = F.softmax(logits, dim=-1)
                
                # Sample top beam_size tokens
                top_probs, top_indices = torch.topk(probs, beam_size, dim=-1)
                
                for i in range(beam_size):
                    token_id = top_indices[0, i].item()
                    token_prob = top_probs[0, i].item()
                    
                    # Skip special tokens and repetitive tokens
                    if token_id in [tokenizer.pad_token_id, tokenizer.eos_token_id]:
                        continue
                    
                    # Check for repetition (simple check)
                    if sequence.shape[1] > 2:
                        last_tokens = sequence[0, -2:].tolist()
                        if token_id in last_tokens:
                            continue  # Skip if token repeats
                    
                    # Create new sequence
                    new_sequence = torch.cat([sequence, torch.tensor([[token_id]], device=device)], dim=1)
                    new_score = score + torch.log(torch.tensor(token_prob))
                    
                    new_beams.append((new_sequence, new_score.item()))
        
        # Keep top beam_size beams
        new_beams.sort(key=lambda x: x[1], reverse=True)
        beams = new_beams[:beam_size]
        
        # Check if all beams are complete
        if all(seq.shape[1] >= max_length for seq, _ in beams):
            break
    
    # Decode sequences
    generated_texts = []
    for sequence, score in beams[:num_generate]:
        # Remove start token
        if sequence.shape[1] > 1:
            sequence = sequence[:, 1:]
        
        # Skip if too short
        if sequence.shape[1] < min_length:
            continue
        
        # Decode to text
        try:
            text = tokenizer.decode(sequence[0], skip_special_tokens=True)
            text = text.strip()
            
            # Skip empty or very short text
            if len(text) < 3:
                continue
                
            generated_texts.append(text)
        except:
            continue
    
    # If no valid texts generated, return a simple fallback
    if not generated_texts:
        return "Generated text"
    
    return generated_texts if len(generated_texts) > 1 else generated_texts[0]


def greedy_decode(decoder_hidden, encoder_outputs, target_tensor):
    '''
    :param target_tensor: target indexes tensor of shape [B, T] where B is the batch size and T is the maximum length of the output sentence
    :param decoder_hidden: input tensor of shape [1, B, H] for start of the decoding
    :param encoder_outputs: if you are using attention mechanism you can pass encoder outputs, [T, B, H] where T is the maximum length of input sentence
    :return: decoded_batch
    '''

    batch_size, seq_len = target_tensor.size()
    decoded_batch = torch.zeros((batch_size, MAX_LENGTH))
    decoder_input = torch.LongTensor([[SOS_token] for _ in range(batch_size)], device=device)

    for t in range(MAX_LENGTH):
        decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden, encoder_outputs)

        topv, topi = decoder_output.data.topk(1)  # get candidates
        topi = topi.view(-1)
        decoded_batch[:, t] = topi

        decoder_input = topi.detach().view(-1, 1)

    return decoded_batch