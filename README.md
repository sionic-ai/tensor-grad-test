```
import torch
import torch.nn as nn

# 신경망 구조 정의
class SimpleNeuralNet(nn.Module):
    def __init__(self):
        super(SimpleNeuralNet, self).__init__()
        # 입력 레이어에서 은닉 레이어로 (입력 차원 10, 출력 차원 5)
        self.hidden = nn.Linear(10, 5)
        # 은닉 레이어에서 출력 레이어로 (입력 차원 5, 출력 차원 1)
        self.output = nn.Linear(5, 1)
        # 활성화 함수
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.hidden(x)
        x = self.sigmoid(x)
        x = self.output(x)
        x = self.sigmoid(x)
        return x

# 모델 생성
model = SimpleNeuralNet()

# 모델 파라메터 고정 오버라이드
for name, parameter in model.named_parameters():
    #parameter.requires_grad = True
    parameter.requires_grad = False

# 더미 데이터 입력 (배치 크기 1, 입력 차원 10)
input_tensor = torch.randn(1, 10)


# forward 연산
output = model(input_tensor)

print("Forward Output:", output)

print("forward : Gradient on the output layer:", model.output.weight.grad, model.output.weight.requires_grad)

# 손실 함수와 backward 연산
target = torch.tensor([1.0])  # 정답 데이터
loss_function = nn.MSELoss()
loss = loss_function(output, target)

# Backward
loss.backward()
print("backward : Gradient on the output layer:", model.output.weight.grad, model.output.weight.requires_grad)

```

```
Forward Output: tensor([[0.5051]], grad_fn=<SigmoidBackward0>)
forward : Gradient on the output layer: None True
backward : Gradient on the output layer: tensor([[-0.0954, -0.1553, -0.1599, -0.1781, -0.0790]]) True
```
