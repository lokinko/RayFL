from transformers import GPT2LMHeadModel, GPT2Tokenizer, GPT2Config
import torch.nn as nn
import torch

class CustomGPT2LMHeadModel(GPT2LMHeadModel):
    def __init__(self, config, condational_method):
        super().__init__(config)
        self.condational_method = condational_method
        self.condational_embedding = nn.Embedding(1000, config.n_embd)
        # 初始化新参数
        self.init_weights()
        
    def forward(
        self,
        input_ids=None,
        user_ids=None, # new
        past_key_values=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        if input_ids is not None:
            if self.condational_method == 'add':
                inputs_embeds = self.transformer.wte(input_ids)  # 获取原始词嵌入
                # 添加额外的嵌入到原始嵌入上
                condational_embeds = self.condational_embedding(user_ids)
                inputs_embeds = inputs_embeds + condational_embeds

            elif self.condational_method == 'concat':
                # 获取原始词嵌入
                inputs_embeds = self.transformer.wte(input_ids)

                # 获取条件嵌入并扩展维度以匹配batch size
                cond_embeds = self.condational_embedding(user_ids)  # (batch_size, 1, embedding_dim)

                # 将条件嵌入作为CLS token添加到每个输入序列的前面
                inputs_embeds = torch.cat([cond_embeds, inputs_embeds], dim=1)

                # 更新attention_mask以考虑新增的CLS token
                if attention_mask is not None:
                    cls_attention = torch.ones((attention_mask.shape[0], 1), device=attention_mask.device)
                    attention_mask = torch.cat([cls_attention, attention_mask], dim=1)

                # 更新position_ids以考虑新增的CLS token
                if position_ids is None:
                    seq_length_with_cls = input_ids.size(1) + 1  # 加上CLS token
                    position_ids = torch.arange(seq_length_with_cls, dtype=torch.long, device=input_ids.device)
                    position_ids = position_ids.unsqueeze(0).expand(input_ids.size(0), seq_length_with_cls)

        # 调用父类的forward方法，但使用我们修改后的inputs_embeds
        return super().forward(
            input_ids=None,  # 因为我们已经提供了inputs_embeds，所以这里设置为None
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,  # 使用修改后的嵌入
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

class PerCustomGPT2LMHeadModel(GPT2LMHeadModel):
    def __init__(self, config):
        super().__init__(config)
        
        self.personal_embedding = nn.Embedding(config.vocab_size, config.n_embd)
        self.condational_embedding = nn.Embedding(1000, config.n_embd)
        
        # 初始化新参数
        self.init_weights()
        
    def forward(
        self,
        input_ids=None,
        user_ids=None, # new
        past_key_values=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        if input_ids is not None:
            inputs_embeds = self.transformer.wte(input_ids)  # 获取原始词嵌入
            personal_embeds = self.personal_embedding(input_ids)

            # 添加额外的嵌入到原始嵌入上
            condational_embeds = self.condational_embedding(user_ids)
            
            inputs_embeds = inputs_embeds + personal_embeds + condational_embeds
        
        # 调用父类的forward方法，但使用我们修改后的inputs_embeds
        return super().forward(
            input_ids=None,  # 因为我们已经提供了inputs_embeds，所以这里设置为None
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,  # 使用修改后的嵌入
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

if __name__ == "__main__":
    # 创建配置和模型实例
    config = GPT2Config(
            vocab_size=1000,
            n_layer=1,
            n_head=1,
            n_embd=32,
            max_position_embeddings=5,
        )
    model = CustomGPT2LMHeadModel(config)

    # 打印模型结构
    print(model)