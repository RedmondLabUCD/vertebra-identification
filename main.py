from utils.manage_data import prep_data, plot_images_with_points, create_dataset, view_heatmaps, plot_images_with_points_256, gather_boundaries, create_mask, plot_test_images_with_points


if __name__ == '__main__':

    # create_dataset()
    # plot_images_with_points()
    # prep_data()
    plot_test_images_with_points('UNet_LM_CL_2.csv',name='LM2')
    plot_test_images_with_points('UNet_LM_CL_3.csv',name='LM3')
