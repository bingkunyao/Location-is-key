from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model_name = "zwhc/LiK"

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True, torch_dtype=torch.bfloat16).cuda()
prompt = '''
	You are a hardware engineer proficient in Verilog. Below I give you a hardware design description and the corresponding code implementation with an erroneous line.Your task is to identify this erroneous line. Hint: the erroneous line may not conform to the design description, or has logic/functional error. Requirements: You should only output the SPECIFIC erroneous line identified. 
    Design description:
    ```
    {Please act as a professional verilog designer.

        Implement a module of an 8-bit adder with multiple bit-level adders in combinational logic.

        Module name:
            adder_8bit
        Input ports:
            a[7:0]: 8-bit input operand A.
            b[7:0]: 8-bit input operand B.
            cin: Carry-in input.
        Output ports:
            sum[7:0]: 8-bit output representing the sum of A and B.
            cout: Carry-out output.

        Implementation:
        The module utilizes a series of bit-level adders (full adders) to perform the addition operation.

        Give me the complete code.
     }
    ```

    Code implementation:
    ```Verilog
    module adder_8bit(
        input [7:0] a, b,
        input cin,
        output [7:0] sum,
        output cout);
        wire [8:0] c;

        full_adder FA0 (.a(a[0]), .b(b[0]), .cin(cin), .sum(sum[0]), .cout(c[0]));
        full_adder FA1 (.a(a[1]), .b(b[1]), .cin(c[0]), .sum(sum[1]), .cout(c[1]));
        full_adder FA2 (.a(a[2]), .b(b[2]), .cin(c[1]), .sum(sum[2]), .cout(c[2]));
        full_adder FA3 (.a(a[3]), .b(b[3]), .cin(c[2]), .sum(sum[3]), .cout(c[3]));
        full_adder FA4 (.a(a[4]), .b(b[3]), .cin(c[3]), .sum(sum[4]), .cout(c[4]));
        full_adder FA5 (.a(a[5]), .b(b[5]), .cin(c[4]), .sum(sum[5]), .cout(c[5]));
        full_adder FA6 (.a(a[6]), .b(b[6]), .cin(c[5]), .sum(sum[6]), .cout(c[6]));
        full_adder FA7 (.a(a[7]), .b(b[7]), .cin(c[6]), .sum(sum[7]), .cout(c[7]));

        assign cout = c[7];
    endmodule

    module full_adder (input a, b, cin, output sum, cout);
        assign {cout, sum} = a + b + cin;
    endmodule
    ```
'''

messages = [{"role": "user", "content": prompt}]
inputs = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to(model.device)
outputs = model.generate(
    inputs,
    max_new_tokens=128,
    do_sample=True,
    top_p=0.95,
    temperature=0.3,
    num_return_sequences=1,
    eos_token_id=tokenizer.eos_token_id,
)
print(tokenizer.decode(outputs[0][len(inputs[0]) :], skip_special_tokens=True))
