import streamlit as st


def render() -> None:
    st.markdown("### Dataset")
    st.markdown(
        "- Zheng et al., Microsoft Research Asia - "
        "[Microsoft GeoLife GPS Trajectory Dataset](https://www.kaggle.com/datasets/arashnic/microsoft-geolife-gps-trajectory-dataset/data)"
    )

    st.markdown("### Papers")
    st.markdown(
        "- Park, J.S. et al. (2023). *Generative agents: Interactive simulacra of human behavior.* Proceedings of the 36th Annual ACM Symposium on User Interface Software and Technology (UIST).\n"
        "- Dabiri, S. & Heaslip, K. (2018). *Inferring transportation modes from GPS trajectories using a convolutional neural network.* Transportation Research Part C.\n"
        "- Xiao, Z. et al. (2012). *Inferring social ties between users with human location history.* Journal of Ambient Intelligence and Humanized Computing.\n"
        "- Zheng, Y. et al. (2010). *GeoLife: A collaborative social networking service among user, location and trajectory.* ACM Trans. Web.\n"
        "- Zheng, Y. et al. (2008). *Understanding mobility based on GPS data.* UbiComp.\n"
    )
