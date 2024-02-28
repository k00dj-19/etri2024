from torch.utils.tensorboard import SummaryWriter

# 로그 디렉토리 지정
writer = SummaryWriter('runs/experiment_1')

# 데이터 로깅
for n_iter in range(100):
    writer.add_scalar('Loss/train', 0.5 * n_iter, n_iter)
    writer.add_scalar('Loss/test', 0.3 * n_iter, n_iter)

# Writer 객체 닫기
writer.close()
