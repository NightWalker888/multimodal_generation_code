num_steps = 4 # 插值得到中间的 2 张图像
interpolation_weight = np.linspace(0, 1, num_steps) 
for weight in interpolation_weight: 
    interval_latents = (1 - weight) * all_img_latents[0] + 
    weight * all_img_latents[1] 
    dec_img = decode_img_latents(interval_latents)[0]