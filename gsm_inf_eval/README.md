## Evaluate models on GSM-infinite

> \[!NOTE\]
> See main README.md for instructions. These are old instructions from Yilun below, keeping for posterity.

We evaluate models on G2 in the following way.
This is from: https://github.com/Infini-AI-Lab/gsm_infinite

### Initialization

```bash
# Create a conda environment and install dependencies
conda create -n gsm-env
conda activate gsm-env
conda install python=3.12

# Install dependencies
pip install -r requirements.txt

# or
pip install -e .
```

### Evaluation

1. **Configure your setup** by editing `gsm-infinite/config.sh`:

   ```bash
   # Set your API configuration
   backend_type='openai'  # or 'gemini', 'anthropic'

   # Configure model and dataset
   model_name='your_model_name'
   save_name='your_save_name'
   ```

2. **Run evaluation**:

   ```bash
   cd gsm_infinite
   sbatch eval_models.sh
   ```

Results are stored in `gsm_infinite/results`

3. **Preprocess Result**:

After running evaluation, we run preprocess.py to preprocess the result for visualization, and print the overall accuracy.

```bash
python preprocess.py
```

4. **View results** with the interactive dashboard and forward it to local browser:
   ```bash
   streamlit run app.py --server.address 0.0.0.0 --server.port 8502
   # On your local device:
   ssh -L 8502:localhost:8502 net_id@g2-login-01.coecis.cornell.edu
   # Then open http://localhost:8502/ on browser to view the result.
   ```

## Citation

If you use GSM-Infinite in your research, please cite our paper:

```bibtex
@misc{zhou2025gsminfinitellmsbehaveinfinitely,
    title={GSM-Infinite: How Do Your LLMs Behave over Infinitely Increasing Context Length and Reasoning Complexity?},
    author={Yang Zhou and Hongyi Liu and Zhuoming Chen and Yuandong Tian and Beidi Chen},
    year={2025},
    eprint={2502.05252},
    archivePrefix={arXiv},
    primaryClass={cs.CL},
    url={https://arxiv.org/abs/2502.05252},
}
```

<!-- ## License -->

<!-- This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details. -->

## Support

- 🐛 **Issues**: [GitHub Issues](https://github.com/Infini-AI-Lab/gsm_infinite/issues)
- 💬 **Discussions**: [GitHub Discussions](https://github.com/Infini-AI-Lab/gsm_infinite/discussions)
- 📧 **Contact**: [yangzho6@andrew.cmu.edu](mailto:yangzho6@andrew.cmu.edu)

______________________________________________________________________

<div align="center">
Made with ❤️ by the Infini-AI Lab team
</div>
