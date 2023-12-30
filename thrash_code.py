fig, ax = plt.subplots(2,3,figsize=(10, 15))
ax[0,0].imshow(observations_orig_clone[10].cpu().detach().numpy())
ax[0,1].imshow(observations_background_clone[10].cpu().detach().numpy())
ax[0,2].imshow(observations_gray_clone[10].cpu().detach().numpy())
ax[1,0].imshow(stimulations_clone[10].cpu().detach().numpy())
ax[1,1].imshow(phosphenes_clone[10].cpu().detach().numpy())
ax[1,2].imsjow(reconstructions["rgb"][10].cpu().detach().numpy())
plt.show()


